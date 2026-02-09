"""Evaluate a trained LoRA adapter on SST-2 dev set."""
from pathlib import Path
import argparse
import json
import re

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# python post_train_learn/scripts/eval_sentiment.py


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sentiment_3cls.yaml"


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_prompt(text: str) -> str:
    return (
        "You are a helpful assistant for sentiment analysis.\n"
        "Classify the sentiment of the following sentence as positive or negative.\n"
        f"Sentence: {text}\n"
        "Answer:"
    )


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return ""


def compute_output_dir(cfg) -> Path:
    training_cfg = cfg["training"]
    base = training_cfg["output_dir"]
    return ROOT / base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="", help="Path to LoRA adapter dir")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples for quick eval")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    dev_items = load_jsonl(ROOT / data_cfg["dev_file"])
    if args.max_samples > 0:
        dev_items = dev_items[: args.max_samples]

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else compute_output_dir(cfg)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    qlora_cfg = cfg.get("qlora", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )

    model_name = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    correct = 0
    total = 0
    for item in dev_items:
        text = item["text"]
        gold = item["label"]
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = normalize_label(decoded)
        if pred == gold:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
