"""
Model representation and MPS(.gz) loader for MIPcc26 instances.

Uses HiGHS via highspy to parse the MPS file into a simple structure
we can feed to CPU/GPU heuristics.
"""

import gzip
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
from highspy import Highs, HighsStatus

@dataclass
class MipInstance:
    name: str
    sense: str                 # "min" or "max"
    num_rows: int
    num_cols: int
    c: np.ndarray              # objective coefficients, shape (n,)
    A: sp.csr_matrix           # constraint matrix, shape (m, n)
    row_lower: np.ndarray      # shape (m,)
    row_upper: np.ndarray      # shape (m,)
    lb: np.ndarray             # variable lower bounds, shape (n,)
    ub: np.ndarray             # variable upper bounds, shape (n,)
    var_types: List[str]       # 'C' (cont), 'I' (int), 'B' (binary)
    var_names: List[str]       # variable names
    row_names: Optional[List[str]] = None


def _read_mps_with_highs(mps_path: str) -> Highs:
    """Load an MPS file into a Highs object (no optimization)."""
    highs = Highs()
    status = highs.readModel(mps_path)

    # Newer highspy returns a HighsStatus enum
    if isinstance(status, HighsStatus):
        if status not in (HighsStatus.kOk, HighsStatus.kWarning):
            raise RuntimeError(f"HiGHS failed to read MPS file: {mps_path} (status {status})")
    else:
        # Fallback for older integer-returning APIs
        if status != 0:
            raise RuntimeError(f"HiGHS failed to read MPS file: {mps_path} (status {status})")

    return highs


def load_mps_instance(path: str) -> MipInstance:
    """
    Load an MPS or gzipped MPS file into a MipInstance.

    Args:
        path: path to .mps or .mps.gz

    Returns:
        MipInstance with CSR matrix and vectors on CPU.
    """
    # If gzipped, decompress to a temporary .mps
    to_delete = None
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f_in, tempfile.NamedTemporaryFile(
            delete=False, suffix=".mps"
        ) as f_out:
            f_out.write(f_in.read())
            tmp_path = f_out.name
            to_delete = tmp_path
    else:
        tmp_path = path

    highs = _read_mps_with_highs(tmp_path)

    # Clean up temporary file if created
    if to_delete is not None and os.path.exists(to_delete):
        os.unlink(to_delete)

    # ---- NEW: pull LP from HighsModel ----
    model = highs.getModel()   # HighsModel
    lp = model.lp_             # HighsLp

    # Basic sizes
    num_cols = lp.num_col_
    num_rows = lp.num_row_

    # Objective
    c = np.array(lp.col_cost_, dtype=float)

    # Variable bounds
    lb = np.array(lp.col_lower_, dtype=float)
    ub = np.array(lp.col_upper_, dtype=float)

    # Row bounds: row_lower_ <= A x <= row_upper_
    row_lower = np.array(lp.row_lower_, dtype=float)
    row_upper = np.array(lp.row_upper_, dtype=float)

    # Matrix A in CSR
    # HiGHS stores constraint matrix as HighsSparseMatrix in column-wise format
    a = lp.a_matrix_   # HighsSparseMatrix
    starts = np.array(a.start_, dtype=int)    # length = num_cols + 1
    indices = np.array(a.index_, dtype=int)
    values = np.array(a.value_, dtype=float)

    # HiGHS already provides correct column-wise starts, so we use them directly
    indptr_csc = starts  # no extra concatenation

    A_csc = sp.csc_matrix(
        (values, indices, indptr_csc),
        shape=(num_rows, num_cols),
    )
    A = A_csc.tocsr()


    # Sense: 1 = minimize, -1 = maximize
    sense = "min" if highs.getObjectiveSense() == 1 else "max"

    # Variable types: 0 = cont, 1 = integer in integrality_ array
    var_types = []
    for j in range(num_cols):
        col_int = lp.integrality_[j]  # 0 = cont, 1 = integer
        if col_int == 0:
            var_types.append("C")
        else:
            if lb[j] >= 0.0 and ub[j] <= 1.0:
                var_types.append("B")
            else:
                var_types.append("I")

    # Names (optional, but useful for writing solution files)
    var_names = list(lp.col_names_)
    row_names = list(lp.row_names_)

    inst = MipInstance(
        name=lp.model_name_,
        sense=sense,
        num_rows=num_rows,
        num_cols=num_cols,
        c=c,
        A=A,
        row_lower=row_lower,
        row_upper=row_upper,
        lb=lb,
        ub=ub,
        var_types=var_types,
        var_names=var_names,
        row_names=row_names,
    )

    return inst
