#!/usr/bin/env bash
# Wrapper script required by MIPcc26
# Usage:
#   ./run_heuristic.sh <instance_path.mps.gz> <output_dir>

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <instance_path.mps.gz> <output_dir>"
  exit 1
fi

INSTANCE_PATH="$1"
OUTPUT_DIR="$2"

# Load modules and env on ARC
module load Miniconda3
source activate mipcc26

# Call the Python entrypoint
python -m src.main "$INSTANCE_PATH" "$OUTPUT_DIR"