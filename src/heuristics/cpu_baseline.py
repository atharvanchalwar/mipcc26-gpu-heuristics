"""
Simple CPU baseline heuristics for MIPcc26.

These are intentionally naive and will later be replaced/augmented by
GPU-accelerated heuristics. The goal is to have a clear reference that:

  - takes a MipInstance
  - builds an integer/binary solution
  - attempts a crude greedy repair of linear constraints
  - exposes a violation metric aligned with the competition tolerances.

We use:
  row_lower <= A x <= row_upper

Violation per row i:
  v_i = max(row_lower[i] - activity_i, 0) + max(activity_i - row_upper[i], 0)
"""

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from ..model_representation import MipInstance


def _initial_integer_solution(inst: MipInstance, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a naive initial solution:

      - Continuous vars ('C'):
          midpoint between lb and ub if ub is finite, else lb
      - Integer/Binary vars ('I', 'B'):
          random integer in [ceil(lb), floor(ub)] if both finite;
          otherwise a small bounded range around lb.

    Returns:
        x: np.ndarray of shape (num_cols,)
    """
    n = inst.num_cols
    x = np.zeros(n, dtype=float)

    lb = inst.lb
    ub = inst.ub

    for j, vtype in enumerate(inst.var_types):
        lo = lb[j]
        hi = ub[j]

        if vtype == "C":
            if np.isfinite(lo) and np.isfinite(hi):
                x[j] = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                x[j] = lo
            elif np.isfinite(hi):
                x[j] = hi
            else:
                x[j] = 0.0
        else:
            # Integer or binary
            if np.isfinite(lo):
                low = int(np.ceil(lo))
            else:
                low = -10
            if np.isfinite(hi):
                high = int(np.floor(hi))
            else:
                high = low + 10

            if high < low:
                high = low

            val = rng.integers(low, high + 1)
            if vtype == "B":
                # Clip to {0,1} for binaries
                val = 1 if val >= 1 else 0
            x[j] = float(val)

    return x


def constraint_violation(
    inst: MipInstance, x: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Compute total violation, max per-row violation, and per-row violation for

        row_lower <= A x <= row_upper.

    Row violation:
        v_i = max(row_lower[i] - activity_i, 0) + max(activity_i - row_upper[i], 0).

    Returns:
        total_violation: sum_i v_i
        max_violation: max_i v_i
        row_violation: np.ndarray shape (m,)
    """
    A: csr_matrix = inst.A
    activity = A.dot(x)  # shape (m,)

    lower_violation = np.maximum(inst.row_lower - activity, 0.0)
    upper_violation = np.maximum(activity - inst.row_upper, 0.0)
    row_violation = lower_violation + upper_violation

    total_violation = float(np.sum(row_violation))
    max_violation = float(np.max(row_violation)) if row_violation.size > 0 else 0.0
    return total_violation, max_violation, row_violation


def greedy_repair(
    inst: MipInstance,
    x: np.ndarray,
    max_sweeps: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Extremely naive greedy repair:

      - For up to `max_sweeps` sweeps:
          * compute violations
          * for each violated row i:
              - pick j with largest |A_ij|
              - compute delta to move activity_i toward the nearest bound
              - update x_j within [lb_j, ub_j], rounding for int/binary

    This is not guaranteed to converge, but should reduce violations.
    """
    A: csr_matrix = inst.A
    lb = inst.lb
    ub = inst.ub
    x = x.copy()

    m = inst.num_rows

    for sweep in range(max_sweeps):
        total_v, max_v, row_v = constraint_violation(inst, x)
        if max_v <= tol:
            # Good enough wrt competition-like tolerance
            break

        # Indices of rows with significant violation
        violated_rows = np.where(row_v > tol)[0]
        if violated_rows.size == 0:
            break

        for i in violated_rows:
            # Access row i of A in CSR
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            idx = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            if idx.size == 0:
                continue

            activity_i = float(np.dot(vals, x[idx]))
            low_i = inst.row_lower[i]
            up_i = inst.row_upper[i]

            # Determine which bound is violated
            delta_needed = 0.0
            if np.isfinite(low_i) and activity_i < low_i - tol:
                # Too low: push up to low_i
                delta_needed = low_i - activity_i
            elif np.isfinite(up_i) and activity_i > up_i + tol:
                # Too high: pull down to up_i
                delta_needed = up_i - activity_i
            else:
                continue  # already within bounds or no finite bound to enforce

            # Choose variable with largest |A_ij|
            j_best = int(idx[np.argmax(np.abs(vals))])
            a_ij = float(A[i, j_best])
            if abs(a_ij) < 1e-12:
                continue

            x_new = x[j_best] + delta_needed / a_ij

            # Respect variable bounds
            lo = lb[j_best]
            hi = ub[j_best]
            if np.isfinite(lo):
                x_new = max(lo, x_new)
            if np.isfinite(hi):
                x_new = min(hi, x_new)

            # Round integer/binary vars
            vtype = inst.var_types[j_best]
            if vtype in ("I", "B"):
                x_new = round(x_new)
                if vtype == "B":
                    x_new = 1.0 if x_new >= 1.0 else 0.0

            x[j_best] = x_new

    return x


def build_cpu_baseline_solution(
    inst: MipInstance,
    seed: int = 0,
    max_sweeps: int = 50,
) -> np.ndarray:
    """
    Public entrypoint for the CPU baseline:

      1. Build a random integer/binary initial solution.
      2. Run greedy repair for up to `max_sweeps` sweeps.

    Returns:
        x: np.ndarray (num_cols,) candidate solution.
    """
    rng = np.random.default_rng(seed)
    x0 = _initial_integer_solution(inst, rng)
    x_rep = greedy_repair(inst, x0, max_sweeps=max_sweeps)
    return x_rep
