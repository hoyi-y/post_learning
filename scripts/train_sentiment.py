"""QLoRA fine-tuning for SST-2 (binary sentiment) using Qwen2-1.5B-Instruct."""
from pathlib import Path
import json
import random
from typing import Dict, List
import argparse
import re

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sentiment_3cls.yaml"


def load_jsonl(path: Path) -> List[Dict]:
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


def tokenize_batch(batch, tokenizer, max_length: int):
    prompts = [build_prompt(t) for t in batch["text"]]
    labels = batch["label"]

    prompt_ids = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    full_texts = [p + " " + l for p, l in zip(prompts, labels)]
    full_ids = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    # Mask prompt part so loss is only on label tokens
    labels_ids = []
    for i in range(len(input_ids)):
        prompt_len = len(prompt_ids["input_ids"][i])
        ids = input_ids[i]
        label = [-100] * prompt_len + ids[prompt_len:]
        labels_ids.append(label)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_ids,
    }


def collate_fn(features, tokenizer):
    # Pad inputs
    batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features],
         "attention_mask": [f["attention_mask"] for f in features]},
        padding=True,
        return_tensors="pt",
    )
    # Pad labels to same length with -100
    max_len = batch["input_ids"].shape[1]
    labels = []
    for f in features:
        lab = f["labels"]
        if len(lab) < max_len:
            lab = lab + [-100] * (max_len - len(lab))
        labels.append(lab)
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume_from", type=str, default="", help="Resume from a specific checkpoint path")
    cli_args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    random.seed(cfg.get("seed", 42))

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora_cfg.get("lora_r", 16),
        lora_alpha=qlora_cfg.get("lora_alpha", 32),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=qlora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, lora_config)

    data_cfg = cfg["data"]
    train_items = load_jsonl(ROOT / data_cfg["train_file"])
    dev_items = load_jsonl(ROOT / data_cfg["dev_file"])

    train_ds = Dataset.from_list(train_items)
    dev_ds = Dataset.from_list(dev_items)

    train_ds = train_ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    dev_ds = dev_ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=dev_ds.column_names,
    )

    data_collator = lambda features: collate_fn(features, tokenizer)

    training_cfg = cfg["training"]
    # Ensure numeric fields are proper types (YAML or env overrides can turn them into strings)
    def _to_int(v):
        return int(v) if isinstance(v, str) else v

    def _to_float(v):
        return float(v) if isinstance(v, str) else v

    training_cfg = {
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
    args_kwargs = dict(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        learning_rate=training_cfg["learning_rate"],
        num_train_epochs=training_cfg["num_train_epochs"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        logging_steps=training_cfg["logging_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=training_cfg.get("dataloader_pin_memory", True),
        report_to=[],
    )

    # Backward compatibility with older transformers
    import inspect
    valid_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    args_kwargs = {k: v for k, v in args_kwargs.items() if k in valid_keys}

    training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    resume_path = None
    if cli_args.resume_from:
        resume_path = cli_args.resume_from
    elif cli_args.resume:
        output_dir = Path(training_cfg["output_dir"])
        if output_dir.exists():
            checkpoints = list(output_dir.glob("checkpoint-*"))
            if checkpoints:
                def _step(p: Path):
                    m = re.search(r"checkpoint-(\d+)", p.name)
                    return int(m.group(1)) if m else -1

                checkpoints.sort(key=_step, reverse=True)
                resume_path = str(checkpoints[0])

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()
    trainer.save_model(training_cfg["output_dir"])


if __name__ == "__main__":
    main()
