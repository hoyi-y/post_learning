# post_train_learn

Lightweight post-training workspace for using pretrained models.

## Structure
- `configs/` experiment configs (YAML/TOML/JSON)
- `logs/` training/inference logs
- `models/` model artifacts (checkpoints)
- `outputs/` predictions/metrics/exports
- `scripts/` runnable scripts (entrypoints)
- `src/` python package

## Conventions
- Scripts are small entrypoints. Core logic lives in `src/post_train_learn/`.
- Artifacts are versioned in `models/` and `outputs/`.

## Setup On Another GPU Machine
These steps assume Linux + conda and an NVIDIA GPU.

1. Clone the repo and enter it.
```bash
git clone <your-repo-url>
cd study_llm/post_train_learn
```

2. Create the conda environment.
```bash
bash scripts/setup_env.sh
conda activate post_train_learn
```

3. Install CUDA-enabled PyTorch (for NVIDIA GPUs).
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the rest of the dependencies.
```bash
pip install -r requirements.txt
```

5. Download SST-2 and preprocess.
```bash
python scripts/download_sst2.py
```

6. Start training.
```bash
python scripts/train_sentiment.py
```

Notes:
- If `download_sst2.py` fails due to network, set `HF_ENDPOINT` to a mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
- If you want to keep only LoRA weights, the default training script already does that.
