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
