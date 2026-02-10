#!/bin/bash
set -euo pipefail

# Set PYTHONPATH
export PYTHONPATH=src:${PYTHONPATH:-}

# Memory fragmentation fix for heavy VLM inference
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/scripts/run_all_models.py