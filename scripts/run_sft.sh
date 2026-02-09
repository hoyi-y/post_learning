#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT/scripts/prepare_sft_sst2.py"
python "$ROOT/scripts/train_sft.py" "$@"
