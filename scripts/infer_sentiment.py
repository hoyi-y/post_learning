"""Inference for a trained LoRA adapter."""
from pathlib import Path
import argparse
import json

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# python post_train_learn/scripts/infer_sentiment.py --text "这家餐厅味道不错"


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sentiment_3cls.yaml"


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
    return ROOT / f"{base}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--adapter_dir", type=str, default="", help="Path to LoRA adapter dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
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

    prompt = build_prompt(args.text)
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

    print(json.dumps({"text": args.text, "prediction": pred, "raw": decoded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
