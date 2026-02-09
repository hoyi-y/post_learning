"""SFT training for SST-2 instruction data using Qwen2-1.5B-Instruct + QLoRA."""
from pathlib import Path
import json

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

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


def build_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction}\n输入: {input_text}\n回答:"


def _run_dir(cfg) -> Path:
    training_cfg = cfg["training"]
    base = training_cfg["output_dir"]
    return ROOT / f"{base}_sft"


def main():
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

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
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

    train_items = load_jsonl(ROOT / "data" / "processed" / "sft_train.jsonl")
    dev_items = load_jsonl(ROOT / "data" / "processed" / "sft_dev.jsonl")

    def to_text(items):
        out = []
        for it in items:
            prompt = build_prompt(it["instruction"], it["input"])
            response = it["output"]
            out.append({"text": prompt + " " + response})
        return out

    train_ds = Dataset.from_list(to_text(train_items))
    dev_ds = Dataset.from_list(to_text(dev_items))

    training_cfg = cfg["training"]
    output_dir = _run_dir(cfg)
    print("output_dir =", output_dir)

    sft_args = SFTConfig(
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
        max_seq_length=max_length,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()

# Accuracy: 0.9495 (828/872)