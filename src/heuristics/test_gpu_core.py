# src/heuristics/test_gpu_core.py
"""
Quick test for GPU core.

Usage:
    python -m src.heuristics.test_gpu_core data/.../instance_01.mps.gz
"""

import sys
import numpy as np
import torch

from ..model_representation import load_mps_instance
from .gpu_core import build_gpu_mip_data, batched_violation


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.heuristics.test_gpu_core <instance_path.mps.gz>")
        sys.exit(1)

    instance_path = sys.argv[1]
    inst = load_mps_instance(instance_path)

    gpu_data = build_gpu_mip_data(inst)
    device = gpu_data.device
    print(f"[TEST] Using device: {device}")

    B = 4
    n = inst.num_cols
    rng = np.random.default_rng(0)
    X = np.zeros((B, n), dtype=np.float32)

    for j, vtype in enumerate(inst.var_types):
        lo = inst.lb[j]
        hi = inst.ub[j]
        if not np.isfinite(lo):
            lo = -10.0
        if not np.isfinite(hi):
            hi = 10.0
        X[:, j] = rng.uniform(lo, hi, size=B)
        if vtype in ("I", "B"):
            X[:, j] = np.round(X[:, j])

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    viol = batched_violation(gpu_data, X_t)
    print(f"[TEST] Batched total violation: {viol}")


if __name__ == "__main__":
    main()
