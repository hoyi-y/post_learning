"""Prepare SFT instruction-style JSONL from SST-2-style JSONL (text, label)."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "processed"


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def to_sft(items):
    out = []
    for it in items:
        text = it.get("text", "").strip()
        label = it.get("label", "").strip().lower()
        if not text or label not in {"positive", "negative"}:
            continue
        out.append(
            {
                "instruction": "判断下面句子的情感是正面还是负面。",
                "input": text,
                "output": label,
            }
        )
    return out


def main():
    train = load_jsonl(IN_DIR / "train.jsonl")
    dev = load_jsonl(IN_DIR / "dev.jsonl")
    train_out = to_sft(train)
    dev_out = to_sft(dev)
    dump_jsonl(OUT_DIR / "sft_train.jsonl", train_out)
    dump_jsonl(OUT_DIR / "sft_dev.jsonl", dev_out)
    print(f"Wrote {OUT_DIR / 'sft_train.jsonl'}")
    print(f"Wrote {OUT_DIR / 'sft_dev.jsonl'}")


if __name__ == "__main__":
    main()
