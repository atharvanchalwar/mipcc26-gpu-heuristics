# src/heuristics/gpu_feasibility_jump.py
"""
GPU Feasibility-Jump-style heuristic.

This heuristic is complementary to the feasibility pump:

  - We maintain a batch X of integer/continuous candidates on the GPU.
  - At each iteration:
      * Compute L1 row violations and a row-wise sign pattern.
      * Build a weighted gradient that focuses on badly violated rows.
      * Take signed "jump" steps (mostly integer) guided by this gradient.
      * Use per-row weights and restarts to escape stagnation.
      * Allow occasional acceptance of slightly worse moves.

Compared to a pure feasibility pump:
  - We do not use a pump term ||x_cont - x_int||^2; instead we focus on
    aggressive, violation-driven "jumps" in integer space, plus some noise.
  - This tends to explore more diverse integer neighborhoods.

Public API:
    gpu_feasibility_jump(inst, max_iters=..., batch_size=..., ...)

This file depends on:
    - GpuMipData / build_gpu_mip_data / batched_activity from gpu_core
    - _sample_initial_batch from gpu_feasibility_pump
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from .gpu_core import GpuMipData, build_gpu_mip_data, batched_activity
from .gpu_feasibility_pump import _sample_initial_batch


# ---------------------------------------------------------------------------
# Violation + row-signs
# ---------------------------------------------------------------------------

def _violation_and_signs(
    gpu_data: GpuMipData,
    X: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute L1 row violation and row sign pattern s_i for each candidate.

    Args:
        X: (B, n) batch of candidates.

    Returns:
        total_violation: (B,) L1 row violation for each candidate.
        s: (B, m) row signs (-1 for too low, +1 for too high, 0 otherwise).
        row_violation: (B, m) per-row L1 violation.
    """
    device = gpu_data.device
    X = X.to(device)

    lower = gpu_data.row_lower.to(device)
    upper = gpu_data.row_upper.to(device)

    # (B, m)
    activity = batched_activity(gpu_data, X)

    too_low = (activity < lower).float()
    too_high = (activity > upper).float()

    # sign of subgradient of L1 violation
    s = -too_low + too_high  # (B, m)

    lower_violation = torch.clamp(lower - activity, min=0.0)
    upper_violation = torch.clamp(activity - upper, min=0.0)
    row_violation = lower_violation + upper_violation  # (B, m)
    total_violation = row_violation.sum(dim=1)         # (B,)

    return total_violation, s, row_violation


# ---------------------------------------------------------------------------
# Weighted gradient builder
# ---------------------------------------------------------------------------

def _weighted_gradient(
    gpu_data: GpuMipData,
    s: Tensor,
    row_weights: Tensor,
    row_violation: Tensor,
) -> Tensor:
    """
    Compute weighted gradient A^T (s * w * scaled_violation) for a batch.

    Args:
        s: (B, m) row signs.
        row_weights: (m,) nonnegative weights.
        row_violation: (B, m) per-row L1 violation.

    Returns:
        grad: (B, n) weighted subgradient direction.
    """
    device = gpu_data.device
    s = s.to(device)
    row_weights = row_weights.to(device)
    row_violation = row_violation.to(device)

    A = gpu_data.A  # (m, n) sparse tensor on device

    # Normalize violation per row so very large rows don't completely dominate.
    # max_violation: (m,)
    max_violation = row_violation.max(dim=0).values.clamp(min=1e-6)
    # (B, m) in [0, 1]
    scaled_violation = row_violation / max_violation

    # (B, m): sign * weight * scaled_violation
    s_weighted = s * row_weights.view(1, -1) * scaled_violation

    # (B, m) -> (m, B)
    sT = s_weighted.transpose(0, 1)
    gradT = torch.sparse.mm(A.t(), sT)  # (n, B)
    grad = gradT.transpose(0, 1)        # (B, n)

    # Mild normalization to avoid exploding steps:
    grad_norm = grad.abs().mean(dim=1, keepdim=True).clamp(min=1.0)
    grad = grad / grad_norm

    return grad


# ---------------------------------------------------------------------------
# Main GPU feasibility-jump heuristic
# ---------------------------------------------------------------------------

def gpu_feasibility_jump(
    inst: MipInstance,
    max_iters: int = 200,
    batch_size: int = 128,
    step_size: float = 1.0,
    weight_update: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GPU feasibility-jump heuristic.

    Parameters:
        inst          : MipInstance
        max_iters     : max outer iterations
        batch_size    : number of candidates processed in parallel
        step_size     : base step size for integer jumps
        weight_update : intensity of row weight increases on stagnation
        seed          : RNG seed
        verbose       : print progress if True

    Returns:
        best_x        : np.ndarray (n,) best solution found
        best_violation: float      L1 constraint violation of best_x
    """
    gpu_data: GpuMipData = build_gpu_mip_data(inst)
    device = gpu_data.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initial batch (integer + continuous) on GPU
    X = _sample_initial_batch(inst, gpu_data, batch_size=batch_size, seed=seed).to(
        device
    )

    var_types = gpu_data.var_types.to(device)
    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    int_mask = var_types > 0
    bin_mask = var_types == 2

    # Enforce integrality once at the start
    if int_mask.any():
        Xi = X[:, int_mask]
        Xi_rounded = torch.round(Xi)
        bin_mask_local = bin_mask[int_mask]
        if bin_mask_local.any():
            Xi_rounded[:, bin_mask_local] = torch.clamp(
                Xi_rounded[:, bin_mask_local], 0.0, 1.0
            )
        # Clip all integer vars to bounds
        lb_int = lb[int_mask]
        ub_int = ub[int_mask]
        Xi_rounded = torch.max(torch.min(Xi_rounded, ub_int), lb_int)
        X[:, int_mask] = Xi_rounded

    m = gpu_data.row_lower.shape[0]
    row_weights = torch.ones(m, device=device)

    best_violation = float("inf")
    best_x = None
    best_idx_global = 0  # track which batch entry holds the elite

    prev_best_global = float("inf")
    no_improve_iters = 0

    weight_update_interval = 10
    restart_interval = 25
    target_tol = 1e-3
    accept_tolerance = 1e-4
    prob_accept_worse = 0.05

    for it in range(max_iters):
        # ---- Current violation & gradient ----
        total_v_curr, s, row_violation = _violation_and_signs(gpu_data, X)
        grad = _weighted_gradient(gpu_data, s, row_weights, row_violation)

        # Global best tracking
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
            print(f"[GPU-FJ] iter {it:03d}: min total violation = {v_val:.3e}")

        if v_val <= target_tol:
            break

        # ---- Stagnation detection & row-weight updates ----
        if v_val >= prev_best_global - 1e-6:
            # Could track "stagnation" here if needed
            pass
        prev_best_global = min(prev_best_global, v_val)

        if no_improve_iters > 0 and no_improve_iters % weight_update_interval == 0:
            mean_row_violation = row_violation.mean(dim=0)  # (m,)
            max_mean = mean_row_violation.max()
            if max_mean > 0:
                normalized = mean_row_violation / max_mean
                row_weights = row_weights + weight_update * normalized
            if verbose:
                print("[GPU-FJ]   weight update: increased weights on violated rows")

        # If badly stuck, restart half of the batch (but keep row_weights)
        if no_improve_iters > 0 and no_improve_iters % restart_interval == 0:
            if verbose:
                print("[GPU-FJ]   stagnation: restarting half of the batch")

            num_restart = batch_size // 2
            all_idx = torch.arange(batch_size, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]

            if candidates.numel() <= num_restart:
                restart_idx = candidates
            else:
                perm = torch.randperm(candidates.numel(), device=device)[:num_restart]
                restart_idx = candidates[perm]

            X_new = _sample_initial_batch(
                inst, gpu_data, batch_size=restart_idx.numel(), seed=seed + it
            ).to(device)

            # Enforce integrality for the restarted subset
            if int_mask.any():
                Xi_new = X_new[:, int_mask]
                Xi_new_rounded = torch.round(Xi_new)
                bin_mask_local = bin_mask[int_mask]
                if bin_mask_local.any():
                    Xi_new_rounded[:, bin_mask_local] = torch.clamp(
                        Xi_new_rounded[:, bin_mask_local], 0.0, 1.0
                    )
                lb_int = lb[int_mask]
                ub_int = ub[int_mask]
                Xi_new_rounded = torch.max(torch.min(Xi_new_rounded, ub_int), lb_int)
                X_new[:, int_mask] = Xi_new_rounded

            X[restart_idx] = X_new

            # Also explicitly keep elite in the population (defensive)
            if best_x is not None:
                X[best_idx_global] = torch.tensor(
                    best_x, device=device, dtype=X.dtype
                )

        # ---- Propose new batch X_prop (integer + continuous) ----
        X_prop = X.clone()

        # Integer variables: signed jump along grad
        if int_mask.any():
            grad_int = grad[:, int_mask]

            # Basic direction: move against the sign of the gradient
            step_dir = -torch.sign(grad_int)

            # Break ties (zero gradient) with random +/- 1
            zero_mask = step_dir == 0
            if zero_mask.any():
                noise = (torch.rand_like(step_dir) - 0.5) * 2.0
                noise_step = torch.sign(noise)
                step_dir = step_dir + zero_mask * noise_step

            step_dir = torch.clamp(step_dir, -1.0, 1.0)

            Xi = X_prop[:, int_mask]
            Xi_new = Xi + step_size * step_dir

            # Round to integer and clip binaries / bounds
            Xi_new = torch.round(Xi_new)

            bin_mask_local = bin_mask[int_mask]
            if bin_mask_local.any():
                Xi_new[:, bin_mask_local] = torch.clamp(
                    Xi_new[:, bin_mask_local], 0.0, 1.0
                )

            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            Xi_new = torch.max(torch.min(Xi_new, ub_int), lb_int)

            X_prop[:, int_mask] = Xi_new

        # Continuous variables: small gradient descent step
        cont_mask = var_types == 0
        if cont_mask.any():
            Xc = X_prop[:, cont_mask]
            grad_c = grad[:, cont_mask]
            Xc = Xc - 0.1 * step_size * grad_c  # smaller step for continuous vars
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            Xc = torch.max(torch.min(Xc, ub_c), lb_c)
            X_prop[:, cont_mask] = Xc

        # ---- Candidate-wise acceptance ----
        total_v_prop, _, _ = _violation_and_signs(gpu_data, X_prop)

        improved = total_v_prop <= (total_v_curr - accept_tolerance)

        worse_but_close = (total_v_prop > total_v_curr) & (
            total_v_prop <= 1.05 * total_v_curr
        )
        rand_uniform = torch.rand_like(total_v_curr)
        accept_worse = worse_but_close & (rand_uniform < prob_accept_worse)

        accept_mask = improved | accept_worse

        # Ensure at least one candidate moves, but DON'T force-move the elite
        if not bool(accept_mask.any()):
            all_idx = torch.arange(batch_size, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]
            forced_idx = candidates[
                torch.randint(0, candidates.numel(), (1,), device=device)
            ]
            accept_mask[forced_idx] = True

        X[accept_mask] = X_prop[accept_mask]

        # Keep elite exactly equal to best_x (defensive programming)
        if best_x is not None:
            X[best_idx_global] = torch.tensor(best_x, device=device, dtype=X.dtype)

    if best_x is None:
        best_x = gpu_data.lb.detach().cpu().numpy()
        best_violation = 1e30

    return best_x, best_violation
