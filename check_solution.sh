#!/usr/bin/env bash
# Usage:
#   ./check_solution.sh <instance.mps.gz> <solution.sol>
#
# Uses the MIPLIB solchecker with competition tolerances:
#   lin_tol = 1e-6, int_tol = 1e-5
#
# Assumes you're running inside the mipcc26 conda env so CONDA_PREFIX is set.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <instance.mps.gz> <solution.sol>"
  exit 1
fi

INSTANCE="$1"
SOLUTION="$2"

CHECKER_DIR="$(dirname "$0")/checker"
CHECKER_BIN="$CHECKER_DIR/bin/solchecker"

if [ ! -x "$CHECKER_BIN" ]; then
  echo "Error: solchecker binary not found or not executable at $CHECKER_BIN"
  echo "       Run 'cd checker && make' first."
  exit 1
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Error: CONDA_PREFIX not set. Activate your mipcc26 conda env first."
  exit 1
fi

# Prepend conda's lib directory so we pick up the right libstdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

"$CHECKER_BIN" "$INSTANCE" "$SOLUTION" 1e-6 1e-5