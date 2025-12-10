# src/heuristics/gpu_randomized_rounding_repair.py
"""
Parallel randomized rounding + repair heuristic on GPU.

High-level idea:
  - Start from a continuous batch sampled within bounds.
  - Randomized rounding to get integer/binary assignments.
  - Repeatedly:
      * evaluate row violation on the integer batch,
      * use a noisy, normalized gradient-like signal to propose small
        integer moves (±1 or binary flips),
      * keep everything batched and LP-free,
      * accept moves candidate-wise if they help or are only slightly worse.
  - Occasionally restart part of the batch when globally stuck, but never
    touch the current global best (elite).
  - Track a global best candidate over all rounds.

Conceptually: a GPU-parallel, WalkSAT-style local repair with gradient
“hints” and randomness, complementary to gpu_feasibility_pump,
gpu_fixprop_rounding, and gpu_feasibility_jump.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from .gpu_core import GpuMipData, build_gpu_mip_data
from .gpu_feasibility_pump import _sample_initial_batch, _violation_and_gradient


def gpu_randomized_rounding_repair(
    inst: MipInstance,
    max_iters: int = 150,
    batch_size: int = 256,
    step_size: float = 1.0,
    noise_scale: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GPU-parallel randomized rounding + repair heuristic.

    Args:
        inst: MipInstance.
        max_iters: maximum repair iterations.
        batch_size: number of parallel integer candidates.
        step_size: integer step size (+/- step_size for general ints).
        noise_scale: base amplitude of random perturbations in move direction.
        seed: RNG seed.
        verbose: whether to print per-iteration stats.

    Returns:
        best_x: np.ndarray (n,) best integer-rounded candidate.
        best_violation: float, total L1 row violation of best_x.
    """
    gpu_data: GpuMipData = build_gpu_mip_data(inst)
    device = gpu_data.device

    # Global seeding for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Start from a continuous batch, then randomized rounding
    # ------------------------------------------------------------------
    X_cont = _sample_initial_batch(
        inst, gpu_data, batch_size=batch_size, seed=seed
    ).to(device)

    var_types = gpu_data.var_types.to(device)
    int_mask = var_types > 0          # integer or binary
    bin_mask = var_types == 2         # binaries among int_mask
    cont_mask = var_types == 0        # continuous (probably none here, but safe)

    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    # Initial integer batch: randomized rounding around X_cont
    X = X_cont.clone()
    if bool(int_mask.any()):
        Xi = X[:, int_mask]

        # Add uniform noise in [-0.5, 0.5] before rounding to inject randomness
        noise_init = torch.rand_like(Xi) - 0.5
        Xi_rounded = torch.round(Xi + noise_init)

        # Clip binaries to {0, 1}
        bin_mask_local = bin_mask[int_mask]
        if bool(bin_mask_local.any()):
            Xi_rounded[:, bin_mask_local] = torch.clamp(
                Xi_rounded[:, bin_mask_local], 0.0, 1.0
            )

        # Respect int bounds
        lb_int = lb[int_mask]
        ub_int = ub[int_mask]
        Xi_rounded = torch.max(torch.min(Xi_rounded, ub_int), lb_int)

        X[:, int_mask] = Xi_rounded

    # ------------------------------------------------------------------
    # Global elite tracking
    # ------------------------------------------------------------------
    best_violation = float("inf")
    best_x = None
    best_idx_global = 0

    no_improve_iters = 0
    max_no_improve = 40  # stop if no global improvement for this many iters

    # Candidate-wise acceptance parameters
    accept_tolerance = 1e-4
    prob_accept_worse = 0.05  # small chance to accept slightly worse moves
    worse_factor = 1.05       # only accept worse if <= 1.05 * current

    # Restart parameters
    restart_interval = 30     # when stuck this many iters, restart part of batch
    restart_frac = 0.3        # restart this fraction of the batch

    for it in range(max_iters):
        # Anneal noise over time: high early, lower later
        t = it / max(1, max_iters - 1)
        curr_noise_scale = noise_scale * (1.0 - 0.7 * t)  # ~ from noise_scale -> 0.3*noise_scale

        # --------------------------------------------------------------
        # Evaluate violation & gradient at current assignment
        # --------------------------------------------------------------
        total_v_curr, grad = _violation_and_gradient(gpu_data, X)

        # Normalize gradient per candidate to avoid exploding moves
        grad_norm = grad.abs().mean(dim=1, keepdim=True).clamp(min=1.0)
        grad = grad / grad_norm

        # Track global best
        min_v, min_idx = torch.min(total_v_curr, dim=0)
        v_val = float(min_v.item())
        if v_val < best_violation - 1e-9:
            best_violation = v_val
            best_x = X[min_idx].detach().cpu().numpy()
            best_idx_global = int(min_idx.item())
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        if verbose:
            print(f"[GPU-RAND-REP] iter {it:03d}: min total violation = {v_val:.3e}")

        if v_val <= 1e-3:
            # nearly feasible
            break

        if no_improve_iters >= max_no_improve:
            if verbose:
                print(
                    f"[GPU-RAND-REP] no improvement for {no_improve_iters} iters, stopping."
                )
            break

        # --------------------------------------------------------------
        # Periodic diversification: restart a subset of the batch
        # (excluding the elite entry), then re-round integrals.
        # --------------------------------------------------------------
        if no_improve_iters > 0 and (no_improve_iters % restart_interval == 0):
            if verbose:
                print("[GPU-RAND-REP] stagnation: restarting part of the batch")

            B = total_v_curr.shape[0]
            num_restart = max(1, int(B * restart_frac))

            all_idx = torch.arange(B, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]

            if candidates.numel() > 0:
                if candidates.numel() <= num_restart:
                    idx = candidates
                else:
                    perm = torch.randperm(candidates.numel(), device=device)[:num_restart]
                    idx = candidates[perm]

                X_new = _sample_initial_batch(
                    inst, gpu_data, batch_size=idx.numel(), seed=seed + it
                ).to(device)

                # integer rounding for restarted subset
                if bool(int_mask.any()):
                    Xi = X_new[:, int_mask]
                    noise_restart = torch.rand_like(Xi) - 0.5
                    Xi_rounded = torch.round(Xi + noise_restart)

                    bin_mask_local = bin_mask[int_mask]
                    if bool(bin_mask_local.any()):
                        Xi_rounded[:, bin_mask_local] = torch.clamp(
                            Xi_rounded[:, bin_mask_local], 0.0, 1.0
                        )

                    lb_int = lb[int_mask]
                    ub_int = ub[int_mask]
                    Xi_rounded = torch.max(torch.min(Xi_rounded, ub_int), lb_int)

                    X_new[:, int_mask] = Xi_rounded

                X[idx] = X_new

                # Re-enforce elite explicitly
                if best_x is not None:
                    X[best_idx_global] = torch.tensor(
                        best_x, device=device, dtype=X.dtype
                    )

        # --------------------------------------------------------------
        # Propose stochastic repair step X_prop
        # --------------------------------------------------------------
        X_prop = X.clone()

        # Integer / binary variables: gradient + noise-based ±1 steps
        if bool(int_mask.any()):
            Xi = X_prop[:, int_mask]
            grad_int = grad[:, int_mask]

            # Deterministic part: move opposite to gradient sign
            dir_det = -torch.sign(grad_int)

            # Add noise in [-0.5, 0.5] scaled by curr_noise_scale
            noise = curr_noise_scale * (torch.rand_like(dir_det) - 0.5)

            direction = dir_det + noise
            direction = torch.sign(direction)  # {-1, 0, +1}

            Xi_new = Xi + step_size * direction

            # Re-round and clamp
            Xi_new = torch.round(Xi_new)

            bin_mask_local = bin_mask[int_mask]
            if bool(bin_mask_local.any()):
                Xi_new[:, bin_mask_local] = torch.clamp(
                    Xi_new[:, bin_mask_local], 0.0, 1.0
                )

            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            Xi_new = torch.max(torch.min(Xi_new, ub_int), lb_int)

            X_prop[:, int_mask] = Xi_new

        # Continuous vars: small corrective step (if any)
        if bool(cont_mask.any()):
            Xc = X_prop[:, cont_mask]
            grad_c = grad[:, cont_mask]
            Xc = Xc - 0.1 * step_size * grad_c
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            Xc = torch.max(torch.min(Xc, ub_c), lb_c)
            X_prop[:, cont_mask] = Xc

        # --------------------------------------------------------------
        # Candidate-wise acceptance
        # --------------------------------------------------------------
        total_v_prop, _ = _violation_and_gradient(gpu_data, X_prop)

        improved = total_v_prop <= (total_v_curr - accept_tolerance)

        # slightly worse moves may be accepted with small probability,
        # but only if not dramatically worse
        worse_but_close = (total_v_prop > total_v_curr) & (
            total_v_prop <= worse_factor * total_v_curr
        )
        rand_uniform = torch.rand_like(total_v_curr)
        accept_worse = worse_but_close & (rand_uniform < prob_accept_worse)

        accept_mask = improved | accept_worse

        # If nobody wants to move, force a few non-elite candidates to move
        if not bool(accept_mask.any()):
            B = total_v_curr.shape[0]
            all_idx = torch.arange(B, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]
            if candidates.numel() > 0:
                k = max(1, B // 10)
                perm = torch.randperm(candidates.numel(), device=device)[:k]
                forced_idx = candidates[perm]
                accept_mask[forced_idx] = True

        X[accept_mask] = X_prop[accept_mask]

        # Keep elite exactly equal to best_x (defensive)
        if best_x is not None:
            X[best_idx_global] = torch.tensor(best_x, device=device, dtype=X.dtype)

    if best_x is None:
        best_x = gpu_data.lb.detach().cpu().numpy()
        best_violation = 1e30
    else:
        best_violation = float(best_violation)

    return best_x, best_violation
