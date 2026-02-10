#!/bin/bash
# Quick test: run a single model on a few images to verify it works
# Usage: ./test_model.sh <model_id> [num_images]
# Example: ./test_model.sh google/paligemma-3b-mix-224
#          ./test_model.sh google/paligemma-3b-mix-224 5
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python test_model.py "$@"
