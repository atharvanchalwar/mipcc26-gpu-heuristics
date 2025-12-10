#!/usr/bin/env bash
# Usage:
#   ./check_solution.sh <instance.mps.gz> <solution.sol>
#
# Uses the MIPLIB solchecker with competition tolerances:
#   lin_tol = 1e-6, int_tol = 1e-5

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <instance.mps.gz> <solution.sol>"
  exit 1
fi

INSTANCE="$1"
SOLUTION="$2"

CHECKER_DIR="$(dirname "$0")/checker"
CHECKER_BIN="$CHECKER_DIR/bin/solchecker"   # <- note /bin here

if [ ! -x "$CHECKER_BIN" ]; then
  echo "Error: solchecker binary not found or not executable at $CHECKER_BIN"
  echo "       Run 'cd checker && make' first."
  exit 1
fi

"$CHECKER_BIN" "$INSTANCE" "$SOLUTION" 1e-6 1e-5