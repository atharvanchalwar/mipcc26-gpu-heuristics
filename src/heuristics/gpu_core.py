# src/heuristics/gpu_core.py
"""
Core GPU utilities for MIPcc26 heuristics.

- Convert MipInstance to a GPU-friendly representation.
- Batched computation of constraint activity and violation on GPU.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..model_representation import MipInstance
from ..utils.device import get_default_device, to_device


@dataclass
class GpuMipData:
    """
    GPU-side representation of a MIP instance.
    Everything is stored as torch Tensors on a given device.
    """
    device: torch.device
    num_rows: int
    num_cols: int

    # Objective + bounds
    c: Tensor          # (n,)
    lb: Tensor         # (n,)
    ub: Tensor         # (n,)
    row_lower: Tensor  # (m,)
    row_upper: Tensor  # (m,)

    # Matrix A in sparse COO: shape (m, n)
    A: Tensor          # sparse_coo_tensor

    # Variable types as numeric flags: 0=cont, 1=int, 2=bin
    var_types: Tensor  # (n,), int64

    # Var names for writing solutions
    var_names: list


def build_gpu_mip_data(inst: MipInstance, device=None) -> GpuMipData:
    """
    Convert a MipInstance (NumPy + SciPy) into GPU-friendly tensors.
    """
    if device is None:
        device = get_default_device()

    A_csr = inst.A.tocsr()
    m, n = A_csr.shape

    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    row_indices = np.repeat(np.arange(m, dtype=np.int64), np.diff(indptr))
    col_indices = indices.astype(np.int64)
    values = data.astype(np.float32)

    indices_2d = np.vstack([row_indices, col_indices])  # (2, nnz)
    indices_t = torch.as_tensor(indices_2d, dtype=torch.long, device=device)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device)

    A_sparse = torch.sparse_coo_tensor(indices_t, values_t, size=(m, n)).coalesce()

    # Objective + bounds
    c = to_device(inst.c.astype(np.float32), device)
    lb = to_device(inst.lb.astype(np.float32), device)
    ub = to_device(inst.ub.astype(np.float32), device)
    row_lower = to_device(inst.row_lower.astype(np.float32), device)
    row_upper = to_device(inst.row_upper.astype(np.float32), device)

    # Var types -> numeric flags
    type_map = {"C": 0, "I": 1, "B": 2}
    var_type_flags = np.array([type_map[t] for t in inst.var_types], dtype=np.int64)
    var_types = to_device(var_type_flags, device).long()

    return GpuMipData(
        device=device,
        num_rows=m,
        num_cols=n,
        c=c,
        lb=lb,
        ub=ub,
        row_lower=row_lower,
        row_upper=row_upper,
        A=A_sparse,
        var_types=var_types,
        var_names=list(inst.var_names),
    )


def batched_activity(gpu_data: GpuMipData, x: Tensor) -> Tensor:
    """
    Compute constraint activity Ax for a batch of solutions.

    Args:
        x: (B, n) dense tensor on same device.

    Returns:
        activity: (B, m)
    """
    device = gpu_data.device
    x = x.to(device)

    # A: (m, n) sparse; want (B, m) = (B, n) @ A^T
    # PyTorch supports sparse @ dense: (m, n) @ (n, B) -> (m, B)
    A = gpu_data.A
    y = torch.sparse.mm(A, x.t())  # (m, B)
    return y.t()  # (B, m)


def batched_violation(gpu_data: GpuMipData, x: Tensor) -> Tensor:
    """
    Compute total constraint violation for a batch of solutions on GPU.

    Args:
        x: (B, n) candidate solutions.

    Returns:
        total_violation: (B,) where each entry is sum over rows of
                         max(lower - Ax, 0) + max(Ax - upper, 0).
    """
    device = gpu_data.device
    x = x.to(device)

    activity = batched_activity(gpu_data, x)  # (B, m)

    lower = gpu_data.row_lower
    upper = gpu_data.row_upper

    lower_violation = torch.clamp(lower - activity, min=0.0)
    upper_violation = torch.clamp(activity - upper, min=0.0)

    row_violation = lower_violation + upper_violation  # (B, m)
    total_violation = row_violation.sum(dim=1)         # (B,)

    return total_violation
