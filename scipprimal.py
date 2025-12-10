#!/usr/bin/env python3
"""
Run SCIP primal heuristics on a batch of MIPcc26 instances.

- Uses PySCIPOpt to read each instance and solve it.
- Emphasizes primal heuristics:
    * heuristics: AGGRESSIVE
    * presolve: FAST
    * separating: FAST
- Imposes a time limit and node limit (so it's more "heuristic-like" than
  a full B&B solve, but still lets root + early nodes trigger heuristics).
- For each instance, writes:
    out_scip/<instance_stem>/solution_1.sol
    out_scip/<instance_stem>/timing.log

  in the same SOL format as your GPU pipeline:
    =obj= <value>
    <var_name> <value>
"""

import os
import time
import gzip
import shutil

from pyscipopt import Model, SCIP_PARAMSETTING


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

INST_DIR = "data/mipcc26_public/original/instances"

INSTANCES = [
    "instance_37.mps.gz",
    "instance_10.mps.gz",
    "instance_25.mps.gz",
    "instance_23.mps.gz",
    "instance_34.mps.gz",
    "instance_09.mps.gz",
    "instance_22.mps.gz",
    "instance_35.mps.gz",
    "instance_03.mps.gz",
    "instance_38.mps.gz",
    "instance_24.mps.gz",
    "instance_36.mps.gz",
    "instance_19.mps.gz",
    "instance_18.mps.gz",
    "instance_12.mps.gz",
    "instance_01.mps.gz",
    "instance_28.mps.gz",
    "instance_27.mps.gz",
    "instance_21.mps.gz",
]

OUT_ROOT = "out_scip"          # root directory for SCIP runs

TIME_LIMIT = 300.0             # seconds per instance (adjust if you want)
NODE_LIMIT = 50000             # max nodes (heuristic-heavy, not full search)


# -------------------------------------------------------------------
# Helper: read (possibly gzipped) MPS into SCIP
# -------------------------------------------------------------------

def read_mps_into_model(model: Model, path: str) -> float:
    """
    Read an .mps or .mps.gz file into a SCIP model.
    Returns: input_time in seconds.
    """
    t0 = time.time()

    if path.endswith(".gz"):
        # Decompress into a temporary file, let SCIP read the plain .mps
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tmp:
            with gzip.open(path, "rb") as fin:
                shutil.copyfileobj(fin, tmp)
            tmp_path = tmp.name
        model.readProblem(tmp_path)
        # Optional: cleanup temp file if you want
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    else:
        model.readProblem(path)

    t1 = time.time()
    return t1 - t0


# -------------------------------------------------------------------
# Solve one instance with SCIP primal heuristics
# -------------------------------------------------------------------

def solve_with_scip_heuristics(inst_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[SCIP] Solving instance: {inst_path}")
    model = Model()

    # --- Input / problem reading time ---
    input_time = read_mps_into_model(model, inst_path)
    print(f"[SCIP]  Loaded problem in {input_time:.3f}s")

    # --- Parameter settings: heuristic-heavy, not full B&B ---
    # Emphasize heuristics:
    model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
    # Keep presolve and separating relatively fast but not off
    model.setPresolve(SCIP_PARAMSETTING.FAST)
    model.setSeparating(SCIP_PARAMSETTING.FAST)

    # Time & node limits: adjust to your experimental design
    model.setParam("limits/time", TIME_LIMIT)
    model.setParam("limits/nodes", NODE_LIMIT)

    # Optional: allow heuristics already in presolve/root
    # (default is fine, but you could tweak specific heuristic params here)

    # --- Optimize (run SCIP + heuristics) ---
    t_solve0 = time.time()
    model.optimize()
    t_solve1 = time.time()
    solve_time = t_solve1 - t_solve0

    status = model.getStatus()
    print(f"[SCIP]  Status: {status}")
    sol = model.getBestSol()

    if sol is None:
        print("[SCIP]  No primal solution found by SCIP (heuristics + search).")
        # still write a timing.log with "no solution"
        timing_path = os.path.join(out_dir, "timing.log")
        with open(timing_path, "w") as f:
            f.write(f"input\t{input_time:.3f}\n")
            f.write(f"solution_1.sol\t{solve_time:.3f}\n")
            f.write("status\tNOSOLUTION\n")
        return

    obj = model.getSolObjVal(sol)
    print(f"[SCIP]  Best objective (primal): {obj:.8f}")

    # --- Write solution in MIPLIB-style SOL format ---
    sol_path = os.path.join(out_dir, "solution_1.sol")
    with open(sol_path, "w") as f:
        f.write(f"=obj= {obj:.16f}\n")
        for v in model.getVars():
            val = model.getSolVal(sol, v)
            f.write(f"{v.name} {val:.16f}\n")

    # --- Write timing log consistent with your GPU pipeline ---
    timing_path = os.path.join(out_dir, "timing.log")
    with open(timing_path, "w") as f:
        f.write(f"input\t{input_time:.3f}\n")
        f.write(f"solution_1.sol\t{solve_time:.3f}\n")

    print(f"[SCIP]  Wrote solution to {sol_path}")
    print(f"[SCIP]  Wrote timing log to {timing_path}")


# -------------------------------------------------------------------
# Batch driver
# -------------------------------------------------------------------

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for inst_name in INSTANCES:
        inst_path = os.path.join(INST_DIR, inst_name)
        stem = os.path.splitext(os.path.splitext(inst_name)[0])[0]  # "instance_01" from "instance_01.mps.gz"
        out_dir = os.path.join(OUT_ROOT, f"{stem}_scip")

        if not os.path.exists(inst_path):
            print(f"[WARN] Instance file not found: {inst_path} (skipping)")
            continue

        solve_with_scip_heuristics(inst_path, out_dir)


if __name__ == "__main__":
    main()
