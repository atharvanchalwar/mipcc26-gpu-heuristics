# src/experiments/run_all_instances.py
"""
Run heuristics on all MIPcc26 instances and log a CSV summary.

Usage examples (from repo root):

  # GPU-only (on a GPU node)
  python -m src.experiments.run_all_instances \
    --instances data/mipcc26_public/original/instances \
    --mode gpu \
    --output results/mipcc26_summary_gpu.csv

  # CPU-only
  python -m src.experiments.run_all_instances \
    --instances data/mipcc26_public/original/instances \
    --mode cpu \
    --output results/mipcc26_summary_cpu.csv

  # Both CPU baseline and GPU FP (requires GPU)
  python -m src.experiments.run_all_instances \
    --instances data/mipcc26_public/original/instances \
    --mode both \
    --output results/mipcc26_summary_both.csv
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch

from ..model_representation import load_mps_instance
from ..heuristics.cpu_baseline import (
    build_cpu_baseline_solution,
    constraint_violation,
)
from ..heuristics.gpu_feasibility_pump import gpu_feasibility_pump


def run_cpu_baseline(inst, num_seeds: int = 5, max_sweeps: int = 50):
    """
    Run the existing CPU baseline heuristic on a single instance.

    Returns:
        objective, total_violation, max_violation, runtime_sec
    """
    start = time.time()
    x_cpu = build_cpu_baseline_solution(
        inst,
        num_seeds=num_seeds,
        max_sweeps=max_sweeps,
    )
    runtime = time.time() - start

    total_v, max_v, _ = constraint_violation(inst, x_cpu)
    obj = float(inst.c @ x_cpu)
    return obj, float(total_v), float(max_v), runtime


def run_gpu_fp(inst, max_iters: int = 50, batch_size: int = 64,
               step_size: float = 1e-3, seed: int = 0):
    """
    Run the GPU feasibility-pump-inspired heuristic on a single instance.

    Returns:
        objective, total_violation, max_violation, runtime_sec
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but run_gpu_fp was called")

    start = time.time()
    x_gpu, v_gpu = gpu_feasibility_pump(
        inst,
        max_iters=max_iters,
        batch_size=batch_size,
        step_size=step_size,
        seed=seed,
    )
    runtime = time.time() - start

    total_v, max_v, _ = constraint_violation(inst, x_gpu)
    obj = float(inst.c @ x_gpu)
    return obj, float(total_v), float(max_v), runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        type=str,
        required=True,
        help="Directory containing *.mps.gz instances "
             "(e.g., data/mipcc26_public/original/instances)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cpu", "gpu", "both"],
        default="gpu",
        help="Which methods to run: cpu, gpu, or both.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to CSV file for results (will be created/overwritten).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of instances to run (for testing).",
    )
    args = parser.parse_args()

    inst_dir = args.instances
    mode = args.mode
    output_csv = args.output

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Collect instance files
    pattern = os.path.join(inst_dir, "*.mps.gz")
    instance_files = sorted(glob.glob(pattern))
    if not instance_files:
        raise RuntimeError(f"No .mps.gz files found in {inst_dir}")

    if args.limit is not None:
        instance_files = instance_files[: args.limit]

    print(f"[RUN-ALL] Found {len(instance_files)} instances in {inst_dir}")
    print(f"[RUN-ALL] Mode: {mode}")
    print(f"[RUN-ALL] Writing results to: {output_csv}")

    # Open CSV and write header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instance",
                "rows",
                "cols",
                "method",
                "objective",
                "total_violation",
                "max_violation",
                "runtime_sec",
            ]
        )

        for idx, inst_path in enumerate(instance_files, start=1):
            inst_name = os.path.basename(inst_path)
            print(f"\n[RUN-ALL] ({idx}/{len(instance_files)}) {inst_name}")

            # Load instance once
            t0 = time.time()
            inst = load_mps_instance(inst_path)
            load_time = time.time() - t0

            m = inst.num_rows
            n = inst.num_cols
            print(
                f"[RUN-ALL] Loaded instance '{inst.name}' "
                f"({m} rows, {n} cols) in {load_time:.3f}s"
            )

            # CPU baseline
            if mode in ("cpu", "both"):
                print("[RUN-ALL]  Running CPU baseline...")
                obj_cpu, tot_cpu, max_cpu, rt_cpu = run_cpu_baseline(inst)
                print(
                    f"[RUN-ALL]  CPU baseline: obj={obj_cpu:.4e}, "
                    f"total_v={tot_cpu:.3e}, max_v={max_cpu:.3e}, "
                    f"time={rt_cpu:.3f}s"
                )
                writer.writerow(
                    [
                        inst_name,
                        m,
                        n,
                        "cpu_baseline",
                        f"{obj_cpu:.16g}",
                        f"{tot_cpu:.6g}",
                        f"{max_cpu:.6g}",
                        f"{rt_cpu:.6f}",
                    ]
                )
                f.flush()

            # GPU FP
            if mode in ("gpu", "both"):
                if not torch.cuda.is_available():
                    print(
                        "[RUN-ALL]  WARNING: CUDA not available; "
                        "skipping GPU-FP for this run."
                    )
                else:
                    print("[RUN-ALL]  Running GPU feasibility pump...")
                    obj_gpu, tot_gpu, max_gpu, rt_gpu = run_gpu_fp(inst)
                    print(
                        f"[RUN-ALL]  GPU-FP: obj={obj_gpu:.4e}, "
                        f"total_v={tot_gpu:.3e}, max_v={max_gpu:.3e}, "
                        f"time={rt_gpu:.3f}s"
                    )
                    writer.writerow(
                        [
                            inst_name,
                            m,
                            n,
                            "gpu_fp",
                            f"{obj_gpu:.16g}",
                            f"{tot_gpu:.6g}",
                            f"{max_gpu:.6g}",
                            f"{rt_gpu:.6f}",
                        ]
                    )
                    f.flush()


if __name__ == "__main__":
    main()
