# src/heuristics/cpu_local_repair.py

import numpy as np
from typing import Tuple

from .cpu_baseline import constraint_violation
from ..model_representation import MipInstance


def cpu_hard_local_repair(
    inst: MipInstance,
    x0: np.ndarray,
    max_passes: int = 5,
    max_flips_per_pass: int = 2000,
    candidate_fraction: float = 0.4,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Deterministic CPU local-repair focused on constraint violation.

    - Starts from integer solution x0.
    - Only touches integer/binary variables (inst.var_types in {'B','I'}).
    - For binaries: single-variable flips x[j] -> 1 - x[j].
    - For general integers: try +/-1 moves (clamped to bounds) and keep the
      best improving move, if any.
    - Restricts attention to a random subset of integer variables each pass
      for speed.
    - Uses a lexicographic improvement rule:

        1) Prefer moves that reduce the maximum row violation (max_v).
        2) If max_v is unchanged (within 1e-9), prefer moves that reduce
           the total violation.

    Returns:
        x_best: np.ndarray, best integer solution found
        best_total_v: float, sum of violations (as in constraint_violation)
        best_max_v: float, max row violation
    """

    rng = np.random.default_rng(seed)

    # Ensure we work with a float copy
    x_best = np.asarray(x0, dtype=float).copy()
    best_total_v, best_max_v, _ = constraint_violation(inst, x_best)

    if verbose:
        print(
            f"[CPU-LOCAL-REPAIR] start: total_violation = {best_total_v:.3e}, "
            f"max_row_violation = {best_max_v:.3e}"
        )

    # Determine which variables are integer / binary
    # inst.var_types is expected to be a sequence like ['B','B','C',...]
    try:
        vtypes = np.array(inst.var_types)
    except AttributeError:
        # Fallback: if var_types not present, assume all variables are binary
        vtypes = np.array(['B'] * inst.num_cols)

    int_mask = np.isin(vtypes, ['B', 'I'])
    int_indices = np.nonzero(int_mask)[0]

    if int_indices.size == 0:
        if verbose:
            print("[CPU-LOCAL-REPAIR] no integer variables; nothing to repair.")
        return x_best, best_total_v, best_max_v

    n_int = int_indices.size
    n_candidates = max(1, int(candidate_fraction * n_int))

    # Bounds (if available) for integer variables
    lb = getattr(inst, "lb", None)
    ub = getattr(inst, "ub", None)

    for p in range(max_passes):
        # Early exit if essentially feasible
        if best_max_v <= 1e-6:
            if verbose:
                print(
                    "[CPU-LOCAL-REPAIR] max violation ~ 0; stopping early."
                )
            break

        improved = False

        # Random subset of integer variable indices
        chosen = rng.choice(int_indices, size=min(n_candidates, n_int), replace=False)

        if verbose:
            print(
                f"[CPU-LOCAL-REPAIR] pass {p:02d}: "
                f"trying up to {max_flips_per_pass} flips over {chosen.size} integer vars"
            )

        flips_done = 0

        # First-improvement strategy over the sampled integer variables
        for j in chosen:
            if flips_done >= max_flips_per_pass:
                break

            vtype = vtypes[j]

            # -----------------------------
            # Case 1: binary variable
            # -----------------------------
            if vtype == 'B':
                new_x = x_best.copy()
                new_x[j] = 1.0 - new_x[j]

                # Enforce bounds if present
                if lb is not None and ub is not None:
                    new_x[j] = min(max(new_x[j], lb[j]), ub[j])

                new_total_v, new_max_v, _ = constraint_violation(inst, new_x)

                # Lexicographic improvement: (max_v, total_v)
                improve_max = new_max_v < best_max_v - 1e-9
                same_max = abs(new_max_v - best_max_v) <= 1e-9
                improve_total = new_total_v < best_total_v - 1e-9

                if improve_max or (same_max and improve_total):
                    if verbose:
                        print(
                            f"[CPU-LOCAL-REPAIR]  flip bin var {j}: "
                            f"{best_total_v:.3e}/{best_max_v:.3e} "
                            f"-> {new_total_v:.3e}/{new_max_v:.3e}"
                        )

                    x_best = new_x
                    best_total_v = new_total_v
                    best_max_v = new_max_v
                    flips_done += 1
                    improved = True

            # -----------------------------
            # Case 2: general integer variable
            # -----------------------------
            elif vtype == 'I':
                base_val = x_best[j]

                candidate_best_total = None
                candidate_best_max = None
                candidate_best_x = None

                # Try +1 and -1 moves
                for delta in (+1.0, -1.0):
                    new_x = x_best.copy()
                    new_val = base_val + delta

                    if lb is not None and ub is not None:
                        new_val = min(max(new_val, lb[j]), ub[j])

                    new_x[j] = new_val

                    new_total_v, new_max_v, _ = constraint_violation(inst, new_x)

                    # Compare candidate against current global best
                    improve_max = new_max_v < best_max_v - 1e-9
                    same_max = abs(new_max_v - best_max_v) <= 1e-9
                    improve_total = new_total_v < best_total_v - 1e-9

                    if improve_max or (same_max and improve_total):
                        # Among the two deltas, keep the better candidate
                        if candidate_best_x is None:
                            candidate_best_x = new_x
                            candidate_best_total = new_total_v
                            candidate_best_max = new_max_v
                        else:
                            # Compare candidates lexicographically
                            c_improve_max = new_max_v < candidate_best_max - 1e-9
                            c_same_max = abs(new_max_v - candidate_best_max) <= 1e-9
                            c_improve_total = new_total_v < candidate_best_total - 1e-9

                            if c_improve_max or (c_same_max and c_improve_total):
                                candidate_best_x = new_x
                                candidate_best_total = new_total_v
                                candidate_best_max = new_max_v

                # If we found an improving +/-1 move for this integer var, accept it
                if candidate_best_x is not None:
                    if verbose:
                        print(
                            f"[CPU-LOCAL-REPAIR]  move int var {j}: "
                            f"{best_total_v:.3e}/{best_max_v:.3e} "
                            f"-> {candidate_best_total:.3e}/{candidate_best_max:.3e}"
                        )

                    x_best = candidate_best_x
                    best_total_v = candidate_best_total
                    best_max_v = candidate_best_max
                    flips_done += 1
                    improved = True

            # If vtype is continuous or unknown, we skip it (shouldn't happen for int_indices)

        if not improved:
            if verbose:
                print(
                    "[CPU-LOCAL-REPAIR] no improving single flip found; "
                    "reached local minimum."
                )
            break

    if verbose:
        print(
            f"[CPU-LOCAL-REPAIR] end: total_violation = {best_total_v:.3e}, "
            f"max_row_violation = {best_max_v:.3e}"
        )

    return x_best, best_total_v, best_max_v
