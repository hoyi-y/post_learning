"""Download SST-2 via HuggingFace datasets and export to JSONL.

Outputs:
  data/processed/train.jsonl
  data/processed/dev.jsonl
  data/processed/test.jsonl
"""
from pathlib import Path
import json

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"

LABEL_MAP = {0: "negative", 1: "positive"}


def dump_split(ds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in ds:
            text = (item.get("sentence") or "").strip()
            label = LABEL_MAP.get(item.get("label"))
            if not text or label is None:
                continue
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


def main():
    ds = load_dataset("glue", "sst2")
    dump_split(ds["train"], OUT_DIR / "train.jsonl")
    dump_split(ds["validation"], OUT_DIR / "dev.jsonl")

    # GLUE/SST-2 test split has no labels; we still export text-only if present.
    test_path = OUT_DIR / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for item in ds["test"]:
            text = (item.get("sentence") or "").strip()
            if not text:
                continue
            f.write(json.dumps({"text": text, "label": ""}, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
