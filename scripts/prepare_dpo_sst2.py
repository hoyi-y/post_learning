"""Prepare DPO pairs from SST-2-style JSONL (text, label)."""
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


def build_prompt(text: str) -> str:
    return (
        "You are a helpful assistant for sentiment analysis.\n"
        "Classify the sentiment of the following sentence as positive or negative.\n"
        f"Sentence: {text}\n"
        "Answer:"
    )


def to_dpo(items):
    out = []
    for it in items:
        text = it.get("text"or"").strip()
        label = it.get("label"or "").strip().lower()
        if not text or label not in {"positive", "negative"}:
            continue
        prompt = build_prompt(text)
        if label == "positive":
            chosen, rejected = "positive", "negative"
        else:
            chosen, rejected = "negative", "positive"
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


def main():
    train = load_jsonl(IN_DIR / "train.jsonl")
    dev = load_jsonl(IN_DIR / "dev.jsonl")
    train_out = to_dpo(train)
    dev_out = to_dpo(dev)
    dump_jsonl(OUT_DIR / "dpo_train.jsonl", train_out)
    dump_jsonl(OUT_DIR / "dpo_dev.jsonl", dev_out)
    print(f"Wrote {OUT_DIR / 'dpo_train.jsonl'}")
    print(f"Wrote {OUT_DIR / 'dpo_dev.jsonl'}")


if __name__ == "__main__":
    main()
