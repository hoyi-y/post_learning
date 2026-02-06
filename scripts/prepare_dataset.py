import json
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Template: expects TSV with columns: text, label
# For SST-2, label is 0/1. We map: 0 -> negative, 1 -> positive.
LABEL_MAP = {"0": "negative", "1": "positive", 0: "negative", 1: "positive"}

def tsv_to_jsonl(tsv_path: Path, jsonl_path: Path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("r", encoding="utf-8") as f_in, jsonl_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            text = row.get("text", "").strip()
            raw_label = row.get("label", "")
            label = LABEL_MAP.get(raw_label, str(raw_label).strip())
            if not text or not label:
                continue
            f_out.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


def main():
    # Example file names
    for split in ["train", "dev", "test"]:
        tsv_path = RAW_DIR / f"{split}.tsv"
        jsonl_path = PROC_DIR / f"{split}.jsonl"
        if tsv_path.exists():
            tsv_to_jsonl(tsv_path, jsonl_path)
            print(f"Wrote {jsonl_path}")
        else:
            print(f"Missing {tsv_path} (skip)")


if __name__ == "__main__":
    main()
