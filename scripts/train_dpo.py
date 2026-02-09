"""DPO training for SST-2 pairs using Qwen2-1.5B-Instruct + QLoRA."""
from pathlib import Path
import argparse
import json

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

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


def _run_dir(cfg) -> Path:
    training_cfg = cfg["training"]
    base = training_cfg["output_dir"]
    return ROOT / f"{base}_dpo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="", help="Resume adapter dir (optional)")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # Ensure numeric fields are proper types
    training_cfg = cfg["training"]

    def _to_int(v):
        return int(v) if isinstance(v, str) else v

    def _to_float(v):
        return float(v) if isinstance(v, str) else v

    cfg["training"] = {
        **training_cfg,
        "per_device_train_batch_size": _to_int(training_cfg.get("per_device_train_batch_size")),
        "per_device_eval_batch_size": _to_int(training_cfg.get("per_device_eval_batch_size")),
        "learning_rate": _to_float(training_cfg.get("learning_rate")),
        "num_train_epochs": _to_float(training_cfg.get("num_train_epochs")),
        "gradient_accumulation_steps": _to_int(training_cfg.get("gradient_accumulation_steps")),
        "eval_steps": _to_int(training_cfg.get("eval_steps")),
        "save_steps": _to_int(training_cfg.get("save_steps")),
        "logging_steps": _to_int(training_cfg.get("logging_steps")),
    }

    model_name = cfg["model"]["name_or_path"]
    max_length = cfg["model"]["max_length"]

    qlora_cfg = cfg.get("qlora", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=qlora_cfg.get("lora_r", 16),
        lora_alpha=qlora_cfg.get("lora_alpha", 32),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=qlora_cfg.get("target_modules"),
    )
    model = get_peft_model(base_model, lora_config)

    train_items = load_jsonl(ROOT / "data" / "processed" / "dpo_train.jsonl")
    dev_items = load_jsonl(ROOT / "data" / "processed" / "dpo_dev.jsonl")

    def _clean(items):
        out = []
        for ex in items:
            prompt = (ex.get("prompt") or "").strip()
            chosen = (ex.get("chosen") or "").strip()
            rejected = (ex.get("rejected") or "").strip()
            if not prompt or not chosen or not rejected:
                continue
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return out

    train_items = _clean(train_items)
    dev_items = _clean(dev_items)

    train_ds = Dataset.from_list(train_items)
    dev_ds = Dataset.from_list(dev_items)

    def _norm(ex):
        return {
            "prompt": str(ex.get("prompt", "")).strip(),
            "chosen": str(ex.get("chosen", "")).strip(),
            "rejected": str(ex.get("rejected", "")).strip(),
        }

    train_ds = train_ds.map(_norm, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(_norm, remove_columns=dev_ds.column_names)

    train_ds = train_ds.filter(lambda ex: ex["prompt"] and ex["chosen"] and ex["rejected"])
    dev_ds = dev_ds.filter(lambda ex: ex["prompt"] and ex["chosen"] and ex["rejected"])

    training_cfg = cfg["training"]
    output_dir = _run_dir(cfg)
    print("output_dir =", output_dir)

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        num_train_epochs=training_cfg["num_train_epochs"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        logging_steps=training_cfg["logging_steps"],
        bf16=training_cfg.get("bf16", True),
        fp16=training_cfg.get("fp16", False),
        max_length=max_length,
        max_prompt_length=max_length - 16,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
