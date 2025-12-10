#!/usr/bin/env python3
"""
Main entrypoint for MIPcc26 GPU-accelerated primal heuristics.

Usage:
    python -m src.main <instance_path.mps.gz> <output_dir>

Behavior (updated):
    - Reads a gzipped MPS instance via HiGHS-based parser.
    - If CUDA is available:
        * Runs four base GPU heuristics:
            - GPU feasibility pump
            - GPU fix-and-propagate-style rounding
            - GPU Feasibility-Jump-style search
            - GPU randomized rounding + repair
        * Picks the solution with the smallest constraint violation
          using a lexicographic rule (max_row_violation, total_violation).
        * Runs a GPU LNS polish around that best GPU candidate.
        * Runs a CPU hard local-repair (binary flip hill-climbing) around
          the best GPU result to further reduce constraint violation.
        * Optionally runs a CPU greedy-repair polish (existing greedy_repair)
          and keeps it only if it improves feasibility.
    - If CUDA is NOT available:
        * Falls back to a CPU baseline heuristic.
    - Always writes:
        * solution_1.sol  (MIPLIB SOL format with =obj= line)
        * timing.log      (input time + solution time)
"""

import os
import sys
import time
from typing import Tuple

import torch
import numpy as np

from .heuristics.gpu_feasibility_pump import gpu_feasibility_pump
from .heuristics.gpu_fixprop_rounding import gpu_fixprop_rounding
from .heuristics.gpu_feasibility_jump import gpu_feasibility_jump
from .heuristics.gpu_randomized_rounding_repair import (
    gpu_randomized_rounding_repair,
)
from .heuristics.gpu_lns_polish import gpu_lns_polish
from .heuristics.cpu_local_repair import cpu_hard_local_repair

from .model_representation import load_mps_instance, MipInstance
from .heuristics.cpu_baseline import (
    build_cpu_baseline_solution,
    constraint_violation,
    greedy_repair,
)


# ---------------------------------------------------------------------------
# Helper: enforce integrality and bounds consistently everywhere
# ---------------------------------------------------------------------------

def project_to_integral_bounds(inst: MipInstance, x: np.ndarray) -> np.ndarray:
    """
    Return a copy of x with:
      - all integer / binary variables rounded to nearest integer,
      - all variables clipped to [lb, ub].

    This is used before any final violation check and before writing
    a solution to disk, so that we never accidentally output a point
    slightly outside the bounds or with non-integral integer vars.
    """
    x_proj = np.asarray(x, dtype=float).copy()

    var_types = getattr(inst, "var_types", None)
    if var_types is not None:
        # Assume "C" for continuous; everything else is integer/binary
        is_int = np.array([vt != "C" for vt in var_types], dtype=bool)
        if is_int.any():
            x_proj[is_int] = np.round(x_proj[is_int])

    # Clip to bounds
    x_proj = np.clip(x_proj, inst.lb, inst.ub)
    return x_proj


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def read_instance(instance_path: str) -> MipInstance:
    """
    Parse gzipped MPS and build internal representation.
    Also records the input (MPS read) time onto the instance as `_input_time`.
    """
    t0 = time.time()
    print(f"[INFO] Reading instance from: {instance_path}")
    inst = load_mps_instance(instance_path)
    t1 = time.time()
    input_time = t1 - t0
    print(
        f"[INFO] Loaded instance '{inst.name}' "
        f"({inst.num_rows} rows, {inst.num_cols} cols) in {input_time:.3f}s"
    )
    # Stash input time for timing.log
    inst._input_time = input_time
    return inst


# ---------------------------------------------------------------------------
# CPU baseline (no GPU available)
# ---------------------------------------------------------------------------

def cpu_baseline_heuristic(
    inst: MipInstance,
    output_dir: str,
    time_limit: float = 300.0,
    num_tries: int = 5,
):
    """
    CPU baseline heuristic:
      - generate several integer solutions with different random seeds
      - (optionally) run greedy repair for each
      - pick the one with smallest (max_row_violation, total_violation)
      - write solution_1.sol and timing.log
    """
    os.makedirs(output_dir, exist_ok=True)

    input_time = getattr(inst, "_input_time", 0.0)
    algo_start = time.time()

    best_x = None
    best_total_v = float("inf")
    best_max_v = float("inf")

    print(f"[INFO] Baseline: trying {num_tries} random seeds")
    for seed in range(num_tries):
        x_candidate = build_cpu_baseline_solution(inst, seed=seed)
        x_candidate = project_to_integral_bounds(inst, x_candidate)

        total_v, max_v, _ = constraint_violation(inst, x_candidate)
        print(
            f"[INFO]  Seed {seed}: total_violation = {total_v:.3e}, "
            f"max_row_violation = {max_v:.3e}"
        )

        # Lexicographic: prefer smaller max_v, then smaller total_v
        better = (max_v < best_max_v - 1e-9) or (
            abs(max_v - best_max_v) <= 1e-9 and total_v < best_total_v - 1e-9
        )
        if better:
            best_max_v = max_v
            best_total_v = total_v
            best_x = x_candidate

    if best_x is None:
        print("[WARN] Baseline: failed to construct any candidate solution")
        # Fall back to all-lower-bounds just to produce *something*
        best_x = project_to_integral_bounds(inst, inst.lb.copy())
        best_total_v, best_max_v, _ = constraint_violation(inst, best_x)

    print(
        f"[INFO] Baseline (best of {num_tries}): "
        f"total violation = {best_total_v:.3e}, "
        f"max row violation = {best_max_v:.3e}"
    )

    # Final projection for safety (idempotent in practice)
    x_final = project_to_integral_bounds(inst, best_x)
    obj_value = float(inst.c @ x_final)

    # Sanity check on final violation
    final_total_v, final_max_v, _ = constraint_violation(inst, x_final)
    print(
        f"[INFO] Baseline final solution: total_violation = {final_total_v:.3e}, "
        f"max_row_violation = {final_max_v:.3e}"
    )

    solution_filename = os.path.join(output_dir, "solution_1.sol")
    with open(solution_filename, "w") as f:
        f.write(f"=obj= {obj_value:.16f}\n")
        for name, val in zip(inst.var_names, x_final):
            f.write(f"{name} {float(val):.16f}\n")

    elapsed = time.time() - algo_start
    timing_path = os.path.join(output_dir, "timing.log")
    with open(timing_path, "w") as f:
        f.write(f"input\t{input_time:.3f}\n")
        f.write(f"solution_1.sol\t{elapsed:.3f}\n")

    print(f"[INFO] Baseline: wrote solution to {solution_filename}")
    print(f"[INFO] Baseline: wrote timing log to {timing_path}")


# ---------------------------------------------------------------------------
# CPU greedy polish (optional last step)
# ---------------------------------------------------------------------------

def cpu_greedy_polish(
    inst: MipInstance,
    x_init: np.ndarray,
    label: str = "cpu_greedy_polish",
) -> Tuple[np.ndarray, float, float]:
    """
    Final CPU-based greedy repair polish around a given starting solution.

    Uses the existing greedy_repair() and constraint_violation() utilities.

    Args:
        inst: MipInstance
        x_init: starting solution (numpy array)
        label: name for logging

    Returns:
        x_best: np.ndarray (n,) repaired candidate
        total_v: float, L1 total violation
        max_v: float, max row violation
    """
    print(f"[INFO] Running {label} around current best solution")

    x_start = project_to_integral_bounds(inst, x_init)

    t0 = time.time()
    try:
        # Preferred: greedy_repair(inst, x0)
        x_repaired = greedy_repair(inst, x_start)
    except TypeError:
        # Fallback if greedy_repair has extra arguments; adjust if needed.
        x_repaired = greedy_repair(inst, x_start, max_iters=100000)
    t1 = time.time()

    x_repaired = project_to_integral_bounds(inst, x_repaired)
    total_v, max_v, _ = constraint_violation(inst, x_repaired)
    print(
        f"[INFO]  {label}: total violation = {total_v:.3e}, "
        f"max row violation = {max_v:.3e}, time = {t1 - t0:.3f}s"
    )

    return x_repaired, total_v, max_v


# ---------------------------------------------------------------------------
# GPU portfolio pipeline: run all four GPU heuristics and pick best
# ---------------------------------------------------------------------------

def gpu_pipeline(inst: MipInstance, instance_path: str, output_dir: str):
    """
    GPU portfolio pipeline:
      - Runs four base GPU heuristics (if CUDA is available):
          * Feasibility pump
          * Fix-and-propagate-style rounding
          * Feasibility-Jump-style search
          * Randomized rounding + repair
      - Picks the candidate with the smallest constraint violation, using
        a lexicographic rule (max_row_violation, total_violation).
      - Runs a GPU LNS polish step around that best candidate.
      - Runs a CPU hard local-repair (single-variable flip hill-climbing)
        around the best GPU candidate.
      - Optionally runs a CPU greedy-repair polish around the best candidate.
      - Keeps the best overall and writes solution_1.sol and timing.log.

    If CUDA is not available, falls back to CPU baseline heuristic.
    """
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available; falling back to CPU baseline heuristic.")
        cpu_baseline_heuristic(inst, output_dir)
        return

    print("[INFO] CUDA is available: running GPU heuristic portfolio")

    os.makedirs(output_dir, exist_ok=True)
    input_time = getattr(inst, "_input_time", 0.0)
    algo_start = time.time()

    # Heuristic registry: (tag, function, kwargs)
    gpu_heuristics = [
        ("gpu_fp", gpu_feasibility_pump, {}),
        ("gpu_fixprop", gpu_fixprop_rounding, {}),
        ("gpu_fj", gpu_feasibility_jump, {}),
        ("gpu_randrep", gpu_randomized_rounding_repair, {}),
    ]

    best_x = None
    best_total_v = float("inf")
    best_max_v = float("inf")
    best_name = None

    # ---------- Stage 1: run base GPU heuristics ----------
    for name, func, kwargs in gpu_heuristics:
        print(f"[INFO] Running heuristic: {name}")
        t0 = time.time()
        try:
            x_candidate, heuristic_v = func(inst, **kwargs)
        except TypeError:
            # In case signatures differ slightly, fall back to inst-only call.
            x_candidate, heuristic_v = func(inst)

        x_candidate = project_to_integral_bounds(inst, x_candidate)

        total_v, max_v, _ = constraint_violation(inst, x_candidate)
        t1 = time.time()
        print(
            f"[INFO]  {name}: raw heuristic_v = {heuristic_v:.3e}, "
            f"total violation = {total_v:.3e}, max row violation = {max_v:.3e}, "
            f"time = {t1 - t0:.3f}s"
        )

        # Lexicographic: prefer smaller max_v, then smaller total_v
        better = (max_v < best_max_v - 1e-9) or (
            abs(max_v - best_max_v) <= 1e-9 and total_v < best_total_v - 1e-9
        )
        if better:
            best_total_v = total_v
            best_max_v = max_v
            best_x = x_candidate
            best_name = name

    if best_x is None:
        print(
            "[WARN] GPU portfolio: all heuristics failed to produce a candidate. "
            "Falling back to CPU baseline."
        )
        cpu_baseline_heuristic(inst, output_dir)
        return

    print(
        f"[INFO] Best base GPU heuristic: {best_name} "
        f"(total violation = {best_total_v:.3e}, "
        f"max row violation = {best_max_v:.3e})"
    )

    # ---------- Stage 2: GPU LNS polish around best_x ----------
    print("[INFO] Running GPU LNS polish around best candidate")
    t_polish0 = time.time()
    try:
        x_polished, v_polished = gpu_lns_polish(
            inst,
            best_x,
            max_outer_iters=20,
            batch_size=64,
            init_radius=32,
            step_size=1.0,
            seed=123,
            verbose=True,
        )
    except Exception as e:
        print(f"[WARN] GPU LNS polish failed with error: {e}")
        x_polished = None
        v_polished = None
    t_polish1 = time.time()

    if x_polished is not None:
        x_polished = project_to_integral_bounds(inst, x_polished)
        total_v_pol, max_v_pol, _ = constraint_violation(inst, x_polished)
        print(
            f"[INFO]  gpu_lns_polish: raw heuristic_v = {v_polished:.3e}, "
            f"total violation = {total_v_pol:.3e}, "
            f"max row violation = {max_v_pol:.3e}, "
            f"time = {t_polish1 - t_polish0:.3f}s"
        )

        better = (max_v_pol < best_max_v - 1e-9) or (
            abs(max_v_pol - best_max_v) <= 1e-9 and total_v_pol < best_total_v - 1e-9
        )
        if better:
            print(
                "[INFO] LNS polish improved the best solution: "
                f"{best_total_v:.3e} -> {total_v_pol:.3e}"
            )
            best_total_v = total_v_pol
            best_max_v = max_v_pol
            best_x = x_polished
            best_name = f"{best_name}+lns"
        else:
            print(
                "[INFO] LNS polish did not improve the best solution; "
                "keeping base heuristic result."
            )

    # ---------- Stage 3: CPU hard local-repair around best_x ----------
    # Try to drive violation further down via single-variable flips.
    if best_total_v > 1e-6:
        print("[INFO] Running cpu_hard_local_repair around best GPU solution")
        t_hr0 = time.time()
        try:
            x_repaired, total_v_rep, max_v_rep = cpu_hard_local_repair(
                inst,
                best_x,
                max_passes=10,            # more aggressive than default
                max_flips_per_pass=4000,
                candidate_fraction=0.6,
                seed=42,
                verbose=True,
            )
        except Exception as e:
            print(f"[WARN] cpu_hard_local_repair failed with error: {e}")
            x_repaired = None
            total_v_rep = None
            max_v_rep = None
        t_hr1 = time.time()

        if x_repaired is not None and total_v_rep is not None:
            x_repaired = project_to_integral_bounds(inst, x_repaired)
            total_v_rep, max_v_rep, _ = constraint_violation(inst, x_repaired)
            print(
                f"[INFO]  cpu_hard_local_repair: total violation = {total_v_rep:.3e}, "
                f"max row violation = {max_v_rep:.3e}, "
                f"time = {t_hr1 - t_hr0:.3f}s"
            )
            # Accept only if strictly better (lexicographic)
            better = (max_v_rep < best_max_v - 1e-9) or (
                abs(max_v_rep - best_max_v) <= 1e-9
                and total_v_rep < best_total_v - 1e-9
            )
            if better:
                print(
                    "[INFO] cpu_hard_local_repair improved the best solution: "
                    f"{best_total_v:.3e} -> {total_v_rep:.3e}"
                )
                best_total_v = total_v_rep
                best_max_v = max_v_rep
                best_x = x_repaired
                best_name = f"{best_name}+cpu_repair"
            else:
                print(
                    "[INFO] cpu_hard_local_repair did not improve the best solution; "
                    "keeping GPU-based result."
                )

    # ---------- Stage 4: CPU greedy repair polish (optional) ----------
    # Only bother if we still have non-negligible violation.
    if best_total_v > 1e-3:
        x_cpu_pol, total_v_cpu, max_v_cpu = cpu_greedy_polish(
            inst, best_x, label="cpu_greedy_polish"
        )
        # Accept only if it *improves* feasibility (never trade feasibility away).
        better = (max_v_cpu < best_max_v - 1e-9) or (
            abs(max_v_cpu - best_max_v) <= 1e-9
            and total_v_cpu < best_total_v - 1e-9
        )
        if better:
            print(
                "[INFO] CPU greedy polish improved the best solution: "
                f"{best_total_v:.3e} -> {total_v_cpu:.3e}"
            )
            best_total_v = total_v_cpu
            best_max_v = max_v_cpu
            best_x = x_cpu_pol
            best_name = f"{best_name}+cpu"
        else:
            print(
                "[INFO] CPU greedy polish did not improve the best solution; "
                "keeping previous best."
            )

    print(
        f"[INFO] Best heuristic overall: {best_name} "
        f"(total violation = {best_total_v:.3e}, "
        f"max row violation = {best_max_v:.3e})"
    )

    # ---------- Write final solution & timing ----------
    # Final safety: enforce integrality and bounds before writing
    x_final = project_to_integral_bounds(inst, best_x)
    obj_value = float(inst.c @ x_final)

    final_total_v, final_max_v, _ = constraint_violation(inst, x_final)
    print(
        f"[INFO] Final solution: total_violation = {final_total_v:.3e}, "
        f"max_row_violation = {final_max_v:.3e}"
    )

    solution_filename = os.path.join(output_dir, "solution_1.sol")
    with open(solution_filename, "w") as f:
        f.write(f"=obj= {obj_value:.16f}\n")
        for name, val in zip(inst.var_names, x_final):
            f.write(f"{name} {float(val):.16f}\n")

    elapsed = time.time() - algo_start
    timing_path = os.path.join(output_dir, "timing.log")
    with open(timing_path, "w") as f:
        f.write(f"input\t{input_time:.3f}\n")
        f.write(f"solution_1.sol\t{elapsed:.3f}\n")

    print(f"[INFO] GPU portfolio: wrote solution to {solution_filename}")
    print(f"[INFO] GPU portfolio: wrote timing log to {timing_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.main <instance_path.mps.gz> <output_dir>")
        sys.exit(1)

    instance_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(instance_path):
        print(f"[ERROR] Instance file not found: {instance_path}")
        sys.exit(1)

    print(f"[INFO] Instance: {instance_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # Load MPS instance (for internal representation + input time)
    inst = read_instance(instance_path)

    # --- GPU portfolio (or CPU baseline if no CUDA) ---
    gpu_pipeline(inst, instance_path, output_dir)


if __name__ == "__main__":
    main()
