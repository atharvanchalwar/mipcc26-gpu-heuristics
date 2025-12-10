# src/heuristics/gpu_feasibility_pump.py
"""
GPU feasibility-pump-style heuristic (batched, LP-free).

Exports:
  - gpu_feasibility_pump(inst, ...)
  - _sample_initial_batch(inst, gpu_data, batch_size, seed)
  - _violation_and_gradient(gpu_data, X, row_weights=None)

Other GPU heuristics (fixprop, randomized rounding, LNS) import
  _sample_initial_batch and _violation_and_gradient, so the signatures
must remain compatible:

    _sample_initial_batch(inst, gpu_data, batch_size, seed) -> Tensor (B, n)
    _violation_and_gradient(gpu_data, X) -> (total_violation, gradient)

`row_weights` is optional and only used inside the feasibility pump.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from .gpu_core import GpuMipData, build_gpu_mip_data, batched_activity


# ----------------------------------------------------------------------
# Helpers shared with other heuristics
# ----------------------------------------------------------------------

def _sample_initial_batch(
    inst: MipInstance,
    gpu_data: GpuMipData,
    batch_size: int = 128,
    seed: int = 0,
) -> Tensor:
    """
    Sample a batch of starting points within variable bounds.

    - Continuous vars: uniform in [lb, ub].
    - Integer/binary vars: uniform in [lb, ub] but *not* rounded here
      so that rounding behavior can be controlled by the calling heuristic.
    """
    device = gpu_data.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass

    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    n = lb.shape[0]
    # (B, n) in [0, 1]
    U = torch.rand(batch_size, n, device=device)
    X = lb.view(1, -1) + U * (ub - lb).view(1, -1)

    return X


def _violation_and_gradient(
    gpu_data: GpuMipData,
    X: Tensor,
    row_weights: Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute (L1) row violation and a subgradient wrt x.

    Args:
        gpu_data: GpuMipData with A, row_lower, row_upper, device, ...
        X: (B, n) batch of candidate points.
        row_weights: optional (m,) nonnegative weights for rows. If None,
                     uses all-ones (compatible with older call sites).

    Returns:
        total_violation: (B,) L1 row violation for each candidate.
        grad: (B, n) subgradient direction d/dx of sum_i violation_i(x).

    NOTE: This function is used by other heuristics, which call it as
          _violation_and_gradient(gpu_data, X) with only two arguments.
          The third argument is optional precisely to preserve that API.
    """
    device = gpu_data.device
    X = X.to(device)

    A = gpu_data.A  # (m, n) sparse on device
    lower = gpu_data.row_lower.to(device)
    upper = gpu_data.row_upper.to(device)

    # Row activities: (B, m)
    activity = batched_activity(gpu_data, X)

    too_low = (activity < lower).float()
    too_high = (activity > upper).float()

    # sign of subgradient for L1 violation
    s = -too_low + too_high  # (B, m)

    lower_violation = torch.clamp(lower - activity, min=0.0)
    upper_violation = torch.clamp(activity - upper, min=0.0)
    row_violation = lower_violation + upper_violation  # (B, m)

    total_violation = row_violation.sum(dim=1)  # (B,)

    # Row weights (for feasibility pump we may amplify some constraints)
    if row_weights is None:
        w = torch.ones_like(lower)  # (m,)
    else:
        w = row_weights.to(device)

    # Weighted sign pattern: (B, m)
    s_weighted = s * w.view(1, -1)

    # Gradient: A^T (s_weighted)
    sT = s_weighted.transpose(0, 1)          # (m, B)
    gradT = torch.sparse.mm(A.t(), sT)       # (n, B)
    grad = gradT.transpose(0, 1)             # (B, n)

    return total_violation, grad


# ----------------------------------------------------------------------
# Main GPU feasibility pump
# ----------------------------------------------------------------------

def gpu_feasibility_pump(
    inst: MipInstance,
    max_iters: int = 250,
    batch_size: int = 128,
    step_size: float = 1.0,
    randomization_prob: float = 0.10,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GPU feasibility-pump-style heuristic (LP-free, gradient-based).

    Skeleton:
      - Work with a batch of points X (fractional "LP-like" points).
      - At each iteration:
          * Round X to get an integer candidate Y.
          * Evaluate violation on Y and keep the best Y seen so far.
          * Use violation gradient at Y to update X (fractional step).
          * Occasionally randomize some integer coords to escape cycles.

    Returns:
        best_x: np.ndarray (n,) best integer-rounded solution found.
        best_violation: float, its L1 total row violation.
    """
    gpu_data: GpuMipData = build_gpu_mip_data(inst)
    device = gpu_data.device

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass

    # Variable metadata
    var_types = gpu_data.var_types.to(device)  # 0=cont, 1=int, 2=bin
    int_mask = var_types > 0
    bin_mask = var_types == 2
    cont_mask = var_types == 0

    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    # Initial fractional batch
    X = _sample_initial_batch(inst, gpu_data, batch_size=batch_size, seed=seed)

    # Row-weight multipliers for the pump (start uniform)
    m = gpu_data.row_lower.shape[0]
    row_weights = torch.ones(m, device=device)

    best_violation = float("inf")
    best_x = None
    best_idx_global = 0
    no_improve_iters = 0

    # Parameters for row-weight updates & randomization
    weight_update_interval = 15
    accept_tol = 1e-4

    for it in range(max_iters):
        # --------------------------------------------------
        # 1) Round X -> Y (integer candidate batch)
        # --------------------------------------------------
        Y = X.clone()

        if bool(int_mask.any()):
            Yi = Y[:, int_mask]
            Yi_round = torch.round(Yi)

            # clamp binaries to {0,1}
            bin_mask_local = bin_mask[int_mask]
            if bool(bin_mask_local.any()):
                Yi_round[:, bin_mask_local] = torch.clamp(
                    Yi_round[:, bin_mask_local], 0.0, 1.0
                )

            # clamp to integer bounds
            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            Yi_round = torch.max(torch.min(Yi_round, ub_int), lb_int)

            Y[:, int_mask] = Yi_round

        if bool(cont_mask.any()):
            Yc = Y[:, cont_mask]
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            Yc = torch.max(torch.min(Yc, ub_c), lb_c)
            Y[:, cont_mask] = Yc

        # --------------------------------------------------
        # 2) Evaluate violation on Y, track global best
        # --------------------------------------------------
        total_v_Y, grad_Y = _violation_and_gradient(gpu_data, Y, row_weights=row_weights)
        min_v, min_idx = torch.min(total_v_Y, dim=0)
        v_val = float(min_v.item())

        if v_val < best_violation - 1e-9:
            best_violation = v_val
            best_x = Y[min_idx].detach().cpu().numpy()
            best_idx_global = int(min_idx.item())
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        if verbose:
            print(f"[GPU-FP] iter {it:03d}: min total violation = {v_val:.3e}")

        if v_val <= 1e-3:
            # near-feasible point found
            break

        # --------------------------------------------------
        # 3) Row-weight update (emphasize systematically violated rows)
        # --------------------------------------------------
        if no_improve_iters > 0 and (no_improve_iters % weight_update_interval == 0):
            # Approximate mean violation per row over the batch
            # We don't have row_violation here, but we can re-compute cheaply
            A = gpu_data.A
            lower = gpu_data.row_lower.to(device)
            upper = gpu_data.row_upper.to(device)

            activity = batched_activity(gpu_data, Y)  # (B, m)
            lower_violation = torch.clamp(lower - activity, min=0.0)
            upper_violation = torch.clamp(activity - upper, min=0.0)
            row_violation = lower_violation + upper_violation  # (B, m)

            mean_row_violation = row_violation.mean(dim=0)  # (m,)
            max_mean = mean_row_violation.max()
            if max_mean > 0:
                normalized = mean_row_violation / max_mean
                row_weights = row_weights + 0.5 * normalized  # slightly emphasize
            if verbose:
                print("[GPU-FP]   row-weights updated")

        # --------------------------------------------------
        # 4) Fractional update: use grad at Y to produce new X
        # --------------------------------------------------
        X_prop = X.clone()

        # Continuous variables: gradient descent step from Y
        if bool(cont_mask.any()):
            Xc = X_prop[:, cont_mask]
            grad_c = grad_Y[:, cont_mask]
            Xc = Y[:, cont_mask] - 0.2 * step_size * grad_c
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            Xc = torch.max(torch.min(Xc, ub_c), lb_c)
            X_prop[:, cont_mask] = Xc

        # Integer vars: treat as pseudo-continuous in X (will be re-rounded in next FP step)
        if bool(int_mask.any()):
            Xi = X_prop[:, int_mask]
            grad_int = grad_Y[:, int_mask]

            # move opposite to gradient sign
            step_dir = -torch.sign(grad_int)
            zero_mask = step_dir == 0
            if bool(zero_mask.any()):
                noise = (torch.rand_like(step_dir) - 0.5) * 2.0
                step_dir = step_dir + zero_mask * torch.sign(noise)
            step_dir = torch.clamp(step_dir, -1.0, 1.0)

            Xi = Y[:, int_mask] + step_size * step_dir

            # keep them within relaxed integer bounds (no rounding here)
            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            Xi = torch.max(torch.min(Xi, ub_int + 0.5), lb_int - 0.5)

            X_prop[:, int_mask] = Xi

        # --------------------------------------------------
        # 5) Mild randomization to avoid cycles
        # --------------------------------------------------
        if randomization_prob > 0.0 and bool(int_mask.any()):
            B = X_prop.shape[0]
            # random mask for candidates to perturb
            mask_batch = torch.rand(B, device=device) < randomization_prob
            if bool(mask_batch.any()):
                idx_batch = torch.nonzero(mask_batch, as_tuple=False).view(-1)
                # For those, randomly flip a few binary vars / jiggle integer vars
                Xi = X_prop[idx_batch][:, int_mask]

                # random +/- 1 jump
                noise = torch.randint_like(Xi, low=-1, high=2)
                Xi = Xi + noise

                # clamp relaxed integers
                lb_int = lb[int_mask]
                ub_int = ub[int_mask]
                Xi = torch.max(torch.min(Xi, ub_int + 0.5), lb_int - 0.5)

                X_prop[idx_batch][:, int_mask] = Xi

        # --------------------------------------------------
        # 6) Accept X_prop if it does not obviously worsen
        #    the *rounded* batch too much
        # --------------------------------------------------
        # Evaluate violation on rounded version of X_prop
        Y_prop = X_prop.clone()
        if bool(int_mask.any()):
            Yi = Y_prop[:, int_mask]
            Yi_round = torch.round(Yi)
            bin_mask_local = bin_mask[int_mask]
            if bool(bin_mask_local.any()):
                Yi_round[:, bin_mask_local] = torch.clamp(
                    Yi_round[:, bin_mask_local], 0.0, 1.0
                )
            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            Yi_round = torch.max(torch.min(Yi_round, ub_int), lb_int)
            Y_prop[:, int_mask] = Yi_round

        if bool(cont_mask.any()):
            Yc = Y_prop[:, cont_mask]
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            Yc = torch.max(torch.min(Yc, ub_c), lb_c)
            Y_prop[:, cont_mask] = Yc

        total_v_prop, _ = _violation_and_gradient(gpu_data, Y_prop, row_weights=row_weights)

        improved = total_v_prop <= (total_v_Y - accept_tol)
        # allow slightly worse with small probability
        worse_but_close = (total_v_prop > total_v_Y) & (
            total_v_prop <= 1.05 * total_v_Y
        )
        rand_u = torch.rand_like(total_v_Y)
        accept_worse = worse_but_close & (rand_u < 0.03)

        accept_mask = improved | accept_worse

        if not bool(accept_mask.any()):
            # ensure at least one non-elite candidate moves
            B = total_v_Y.shape[0]
            all_idx = torch.arange(B, device=device)
            mask = all_idx != best_idx_global
            candidates = all_idx[mask]
            if candidates.numel() > 0:
                forced = candidates[torch.randint(0, candidates.numel(), (1,), device=device)]
                accept_mask[forced] = True

        X[accept_mask] = X_prop[accept_mask]

        # keep elite equal to best_x defensively
        if best_x is not None:
            X[best_idx_global] = torch.tensor(best_x, device=device, dtype=X.dtype)

    # Fallback if for some reason best_x was never set
    if best_x is None:
        best_x = gpu_data.lb.detach().cpu().numpy()
        best_violation = 1e30

    return best_x, best_violation
