# src/heuristics/gpu_lns_polish.py
"""
GPU-based local neighborhood search (LNS) polish heuristic.

High-level idea:
  - Start from a (near-)feasible mixed-integer solution x_start.
  - On GPU, repeatedly generate a batch of "neighbors" by perturbing
    a small subset of integer variables, guided by a violation gradient.
  - Evaluate constraint violation for all neighbors in parallel.
  - Keep the best candidate globally and shrink the neighborhood when stuck.

This is a lightweight, LP-free polishing step that is intended to be used
AFTER a stronger heuristic (e.g., gpu_feasibility_jump / gpu_feasibility_pump)
has already found a reasonably good solution.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from .gpu_core import GpuMipData, build_gpu_mip_data
from .gpu_feasibility_pump import _violation_and_gradient


def gpu_lns_polish(
    inst: MipInstance,
    x_start: np.ndarray,
    max_outer_iters: int = 20,
    batch_size: int = 64,
    init_radius: int = 32,
    step_size: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GPU-based local neighborhood search (LNS) around x_start.

    Args:
        inst: MipInstance describing the MILP.
        x_start: np.ndarray (n,) starting solution (not necessarily fully feasible).
        max_outer_iters: maximum number of neighborhood iterations.
        batch_size: number of neighbors per iteration.
        init_radius: max number of integer variables considered for flipping.
        step_size: integer step magnitude (usually 1.0 for general ints).
        seed: RNG seed.
        verbose: whether to print progress.

    Returns:
        best_x: np.ndarray (n,) best candidate found (mixed, integers integral).
        best_violation: float, total L1 row violation of best_x.
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

    # --- Prepare starting point on GPU ---
    x0 = torch.as_tensor(x_start, dtype=torch.float32, device=device).clone()

    # Bounds
    lb = gpu_data.lb.to(device)
    ub = gpu_data.ub.to(device)

    # Variable types: 0 = cont, 1 = int, 2 = bin (as in gpu_core)
    var_types = gpu_data.var_types.to(device)
    int_mask = var_types > 0
    bin_mask = var_types == 2
    cont_mask = var_types == 0

    # Enforce integrality and bounds for start
    if bool(int_mask.any()):
        x_int = x0[int_mask]
        x_int = torch.round(x_int)

        # Clamp binaries to {0,1}
        bin_mask_local = bin_mask[int_mask]
        if bool(bin_mask_local.any()):
            x_int[bin_mask_local] = torch.clamp(x_int[bin_mask_local], 0.0, 1.0)

        # Clamp ints to bounds
        lb_int = lb[int_mask]
        ub_int = ub[int_mask]
        x_int = torch.max(torch.min(x_int, ub_int), lb_int)
        x0[int_mask] = x_int

    # Clamp continuous vars too
    if bool(cont_mask.any()):
        x_cont = x0[cont_mask]
        lb_c = lb[cont_mask]
        ub_c = ub[cont_mask]
        x_cont = torch.max(torch.min(x_cont, ub_c), lb_c)
        x0[cont_mask] = x_cont

    # Evaluate starting violation
    X0 = x0.unsqueeze(0)  # (1, n)
    total_v0, _ = _violation_and_gradient(gpu_data, X0)
    best_violation = float(total_v0.item())
    best_x = x0.detach().cpu().numpy()

    if verbose:
        print(f"[GPU-LNS] start: total violation = {best_violation:.3e}")

    # If there are no integer variables, nothing to polish
    if not bool(int_mask.any()):
        if verbose:
            print("[GPU-LNS] no integer vars; nothing to do.")
        return best_x, best_violation

    # Indices of integer variables in full variable space
    int_idx = torch.nonzero(int_mask, as_tuple=False).view(-1)
    if int_idx.numel() == 0:
        if verbose:
            print("[GPU-LNS] int_idx empty; nothing to do.")
        return best_x, best_violation

    # Neighborhood radius in "number of integer variables we consider"
    radius = min(init_radius, int_idx.numel())
    no_improve_outer = 0
    max_no_improve_outer = 5

    for outer in range(max_outer_iters):
        # Early exit if essentially feasible
        if best_violation <= 1e-3:
            if verbose:
                print(
                    f"[GPU-LNS] best_violation ~ 0 (={best_violation:.3e}); stopping."
                )
            break

        # Current center is always global best so far
        center = torch.as_tensor(best_x, dtype=torch.float32, device=device)

        # Evaluate gradient at center (single batch)
        X_center = center.unsqueeze(0)
        total_v_center, grad_center = _violation_and_gradient(gpu_data, X_center)
        v_center = float(total_v_center.item())

        if verbose:
            print(
                f"[GPU-LNS] outer {outer:02d}: center violation = {v_center:.3e}, "
                f"radius = {radius}, batch = {batch_size}"
            )

        # Select top-|grad| integer coords as candidate move positions
        grad_int = grad_center[0, int_mask]  # (n_int,)
        abs_grad_int = grad_int.abs()

        k = min(radius, int_idx.numel())
        if k <= 0:
            if verbose:
                print("[GPU-LNS] radius became 0; stopping.")
            break

        # If gradient is all zeros, choose integers uniformly at random
        if float(abs_grad_int.max().item()) <= 1e-12:
            topk_int_positions = torch.randperm(int_idx.numel(), device=device)[:k]
        else:
            _, topk_int_positions = torch.topk(abs_grad_int, k, largest=True)
        topk_vars = int_idx[topk_int_positions]  # indices in full var space

        if topk_vars.numel() == 0:
            if verbose:
                print("[GPU-LNS] no candidate integer positions to move; stopping.")
            break

        # For each neighbor, pick a small subset of these to move
        moves_per_candidate = min(4, topk_vars.numel())
        choices = torch.randint(
            low=0,
            high=topk_vars.numel(),
            size=(batch_size, moves_per_candidate),
            device=device,
        )  # (B, moves_per_candidate)
        var_choices = topk_vars[choices]  # (B, moves_per_candidate)

        # Direction for each chosen variable, guided by gradient at center
        grad_all = grad_center[0]  # (n,)
        dir_sign = -torch.sign(grad_all[var_choices])  # (B, moves_per_candidate)

        # If gradient is 0, inject random ±1
        zero_mask = dir_sign == 0
        if bool(zero_mask.any()):
            rand_dir = torch.sign(torch.rand_like(dir_sign) - 0.5)
            dir_sign = torch.where(zero_mask, rand_dir, dir_sign)

        # Build neighbor batch from center
        X_neighbors = center.unsqueeze(0).repeat(batch_size, 1)  # (B, n)
        delta = torch.zeros_like(X_neighbors)
        delta.scatter_add_(1, var_choices, step_size * dir_sign)
        X_neighbors = X_neighbors + delta

        # Re-enforce integrality and bounds on neighbors
        if bool(int_mask.any()):
            X_int = X_neighbors[:, int_mask]
            X_int = torch.round(X_int)

            # clamp binaries to {0,1}
            bin_mask_local = bin_mask[int_mask]
            if bool(bin_mask_local.any()):
                X_int[:, bin_mask_local] = torch.clamp(
                    X_int[:, bin_mask_local], 0.0, 1.0
                )

            lb_int = lb[int_mask]
            ub_int = ub[int_mask]
            X_int = torch.max(torch.min(X_int, ub_int), lb_int)
            X_neighbors[:, int_mask] = X_int

        if bool(cont_mask.any()):
            X_c = X_neighbors[:, cont_mask]
            lb_c = lb[cont_mask]
            ub_c = ub[cont_mask]
            X_c = torch.max(torch.min(X_c, ub_c), lb_c)
            X_neighbors[:, cont_mask] = X_c

        # Evaluate all neighbors
        total_v_batch, _ = _violation_and_gradient(gpu_data, X_neighbors)
        min_v, min_idx = torch.min(total_v_batch, dim=0)
        v_best_batch = float(min_v.item())

        if v_best_batch < best_violation - 1e-9:
            # Improved global best
            best_violation = v_best_batch
            best_x = X_neighbors[min_idx].detach().cpu().numpy()
            no_improve_outer = 0
            if verbose:
                print(
                    f"[GPU-LNS]   improved: new best violation = {best_violation:.3e}"
                )
        else:
            no_improve_outer += 1
            # Shrink the neighborhood slightly when stuck
            radius = max(1, int(radius * 0.7))
            if verbose:
                print(
                    f"[GPU-LNS]   no improvement (streak={no_improve_outer}), "
                    f"shrinking radius to {radius}"
                )
            if no_improve_outer >= max_no_improve_outer:
                if verbose:
                    print(
                        f"[GPU-LNS]   no improvement for {no_improve_outer} outer iters, stopping."
                    )
                break

        # If radius has collapsed to 1 and we already failed to improve once,
        # it’s unlikely that further iterations help.
        if radius <= 1 and no_improve_outer > 0:
            if verbose:
                print("[GPU-LNS] radius collapsed to 1 with no improvement, stopping.")
            break

    return best_x, best_violation
