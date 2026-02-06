#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../conda-env.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda or Anaconda first."
  exit 1
fi

conda env create -f "$ENV_FILE"

echo "Environment created. Activate with:"
echo "  conda activate post_train_learn"
echo ""
echo "Install CUDA-enabled PyTorch via pip (cu121):"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "Then install the rest of the deps:"
echo "  pip install -r requirements.txt"
