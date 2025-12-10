# src/heuristics/gpu_fixprop_rounding.py
"""
GPU-based fix-and-propagate-style rounding / repair heuristic.

High-level idea:
  1. Sample a batch of mixed (cont + int) assignments on GPU.
  2. Round/fix all integer/binary variables once (pure rounding start).
  3. Perform a short "repair" phase:
       * Continuous vars: small steps along -grad, then project to bounds.
       * Integer/binary: sign steps of ±1, then round + project to bounds.
  4. Track a global elite (best_x, best_violation) and never lose it.
  5. Use mild randomization + occasional restarts if we stagnate.

Compared to gpu_feasibility_jump:
  - No row weights, no elaborate restarts: this is intentionally simpler.
  - Think of it as: "good rounding + a few strong repair sweeps".
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from .gpu_core import GpuMipData, build_gpu_mip_data
from .gpu_feasibility_pump import _sample_initial_batch, _violation_and_gradient


def gpu_fixprop_rounding(
    inst: MipInstance,
    max_iters: int = 200,
    batch_size: int = 64,
    step_size: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GPU-based fix-and-propagate-style rounding / repair heuristic.

    Integers/binaries are rounded once, then both integer and continuous
    variables are updated via projected, normalized subgradient steps to
    reduce L1 constraint violation.

    Args:
        inst: MipInstance describing the MILP.
        max_iters: max propagation / repair iterations.
        batch_size: number of parallel candidates.
        step_size: base step size (1.0 is natural for integer moves).
        seed: RNG seed for initial batch.
        verbose: whether to print per-iteration stats.

    Returns:
        best_x: np.ndarray (n,) best candidate (ints integral).
        best_violation: float, total row violation of best_x.
    """
    gpu_data: GpuMipData = build_gpu_mip_data(inst)
    device = gpu_data.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Initial mixed batch (continuous + integer) on GPU
    # ------------------------------------------------------------------
    X = _sample_initial_batch(inst, gpu_data, batch_size=batch_size, seed=seed).to(
        device
    )

    # var_types: 0 = continuous, 1 = integer, 2 = binary
    var_types = gpu_data.var_types.to(device)
    int_mask = var_types > 0
    bin_mask = var_types == 2
    cont_mask = var_types == 0

    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    # ---- Fix integer/binary variables once (pure rounding start) ----
    if bool(int_mask.any()):
        X_int = X[:, int_mask]
        X_int_rounded = torch.round(X_int)

        # Clip binaries to {0, 1}
        bin_mask_local = bin_mask[int_mask]
        if bool(bin_mask_local.any()):
            X_int_rounded[:, bin_mask_local] = torch.clamp(
                X_int_rounded[:, bin_mask_local], 0.0, 1.0
            )

        # Respect integer bounds
        lb_int = lb[int_mask]
        ub_int = ub[int_mask]
        X_int_rounded = torch.max(torch.min(X_int_rounded, ub_int), lb_int)

        X[:, int_mask] = X_int_rounded

    # ------------------------------------------------------------------
    # Global elite tracking
    # ------------------------------------------------------------------
    best_violation = float("inf")
    best_x = None
    best_idx_global = 0

    no_improve_iters = 0
    max_no_improve = 30  # stop if totally stuck for this many iterations

    # Acceptance parameters
    accept_tolerance = 1e-4
    prob_accept_worse = 0.05  # small chance to accept slightly worse moves

    # Simple restart to inject diversity if stalled
    restart_interval = 15   # how often to consider a restart (in iters)

    for it in range(max_iters):
        # --------------------------------------------------------------
        # Evaluate violation & gradient at current batch
        # --------------------------------------------------------------
        total_v_curr, grad = _violation_and_gradient(gpu_data, X)

        # Mild normalization to avoid exploding steps:
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
            print(f"[GPU-FIXPROP] iter {it:03d}: min total violation = {v_val:.3e}")

        # Early exit if nearly feasible
        if v_val <= 1e-3:
            break

        # If hopelessly stuck, stop
        if no_improve_iters >= max_no_improve:
            if verbose:
                print(
                    f"[GPU-FIXPROP] no improvement for {no_improve_iters} iters, stopping."
                )
            break

        # --------------------------------------------------------------
        # Occasional simple restart: re-sample a fraction of the batch,
        # but never touch the elite.
        # --------------------------------------------------------------
        if no_improve_iters > 0 and no_improve_iters % restart_interval == 0:
            if verbose:
                print("[GPU-FIXPROP] stagnation: restarting fraction of batch")
            B = total_v_curr.shape[0]
            num_restart = max(1, B // 3)  # restart ~1/3 of the batch

            all_idx = torch.arange(B, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]

            if candidates.numel() > 0:
                if candidates.numel() <= num_restart:
                    restart_idx = candidates
                else:
                    perm = torch.randperm(candidates.numel(), device=device)[:num_restart]
                    restart_idx = candidates[perm]

                X_new = _sample_initial_batch(
                    inst, gpu_data, batch_size=restart_idx.numel(), seed=seed + it
                ).to(device)

                # Enforce integrality for restarted subset
                if bool(int_mask.any()):
                    Xi_new = X_new[:, int_mask]
                    Xi_new_rounded = torch.round(Xi_new)
                    bin_mask_local = bin_mask[int_mask]
                    if bool(bin_mask_local.any()):
                        Xi_new_rounded[:, bin_mask_local] = torch.clamp(
                            Xi_new_rounded[:, bin_mask_local], 0.0, 1.0
                        )
                    lb_int = lb[int_mask]
                    ub_int = ub[int_mask]
                    Xi_new_rounded = torch.max(
                        torch.min(Xi_new_rounded, ub_int), lb_int
                    )
                    X_new[:, int_mask] = Xi_new_rounded

                X[restart_idx] = X_new

                # Re-enforce elite explicitly
                if best_x is not None:
                    X[best_idx_global] = torch.tensor(
                        best_x, device=device, dtype=X.dtype
                    )

        # --------------------------------------------------------------
        # Propose repair step X_prop
        # --------------------------------------------------------------
        X_prop = X.clone()

        # Integer / binary variables: sign step of ±1, then round + clamp
        if bool(int_mask.any()):
            grad_int = grad[:, int_mask]

            # move opposite to gradient sign
            step_dir = -torch.sign(grad_int)

            # if grad is zero, add small random direction to escape flats
            zero_mask = step_dir == 0
            if bool(zero_mask.any()):
                noise = (torch.rand_like(step_dir) - 0.5) * 2.0  # [-1, 1]
                noise_step = torch.sign(noise)
                step_dir = step_dir + zero_mask * noise_step

            # final step in {-1, 0, +1}
            step_dir = torch.clamp(step_dir, -1.0, 1.0)

            Xi = X_prop[:, int_mask]
            Xi_new = Xi + step_size * step_dir

            # round & clamp
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

        # Continuous variables: small corrective step
        if bool(cont_mask.any()):
            Xc = X_prop[:, cont_mask]
            grad_c = grad[:, cont_mask]
            # smaller step in continuous space
            Xc = Xc - 0.2 * step_size * grad_c
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
        # but only if not dramatically worse (<= 1.05 * current)
        worse_but_close = (total_v_prop > total_v_curr) & (
            total_v_prop <= 1.05 * total_v_curr
        )
        rand_uniform = torch.rand_like(total_v_curr)
        accept_worse = worse_but_close & (rand_uniform < prob_accept_worse)

        accept_mask = improved | accept_worse

        # If nobody wants to move, force a small random subset (excluding elite)
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

        # keep elite exactly equal to best_x (defensive)
        if best_x is not None:
            X[best_idx_global] = torch.tensor(best_x, device=device, dtype=X.dtype)

    if best_x is None:
        # fallback
        best_x = gpu_data.lb.detach().cpu().numpy()
        best_violation = 1e30
    else:
        best_violation = float(best_violation)

    return best_x, best_violation
