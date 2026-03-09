#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import asdict

import torch
from datasets import load_dataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-8B (Unsloth) on a JSONL dataset in messages format."
    )

    # Model / data
    p.add_argument("--model_name", type=str,
                   default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                   help="Base model checkpoint.")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to JSONL with {'messages': [...]} examples.")
    p.add_argument("--output_dir", type=str, required=True)

    # Split like notebook
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--eval_frac", type=float, default=0.2)

    # Context / quantization
    p.add_argument("--max_seq_length", type=int, default=16384)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["auto", "float16", "bfloat16"])

    # LoRA settings (notebook defaults)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # What to finetune (notebook: both vision + language + attention + mlp)
    p.add_argument("--finetune_vision_layers", action="store_true", default=True)
    p.add_argument("--finetune_language_layers", action="store_true", default=True)
    p.add_argument("--finetune_attention_modules", action="store_true", default=True)
    p.add_argument("--finetune_mlp_modules", action="store_true", default=True)

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=3407)

    # Optional: cap dataset for quick tests
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None)

    # Saving
    p.add_argument("--save_lora_only", action="store_true", default=False,
                   help="If set, saves adapters only (default TRL save_model still works).")

    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(dtype_str)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = resolve_dtype(args.dtype)

    # ---- Load model ----
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)

    # ---- LoRA ----
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    # ---- Dataset ----
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    n = len(ds)
    n_train = max(1, int(args.train_frac * n))
    n_eval = max(1, int(args.eval_frac * n))
    # same style as notebook: contiguous split
    train_ds = ds.select(range(0, n_train))
    eval_ds = ds.select(range(n_train, min(n_train + n_eval, n)))

    if args.max_train_examples is not None:
        train_ds = train_ds.select(range(0, min(args.max_train_examples, len(train_ds))))
    if args.max_eval_examples is not None:
        eval_ds = eval_ds.select(range(0, min(args.max_eval_examples, len(eval_ds))))

    print(f"Train examples: {len(train_ds)} | Eval examples: {len(eval_ds)}")

    FastVisionModel.for_training(model)

    # ---- Collator ----
    data_collator = UnslothVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # ---- TRL config ----
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",

        # IMPORTANT for vision finetuning (as notebook notes)
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},

        # keep same length behavior
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
    )

    start_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    stats = trainer.train()

    trainer.save_model(args.output_dir)

    # Save tokenizer as well (often useful for repo)
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception:
        pass

    # Print simple memory stats (no widgets)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        max_mem = props.total_memory / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"GPU: {props.name} | Max memory: {max_mem:.2f} GB")
        print(f"Peak reserved memory: {peak_reserved:.2f} GB")
        if start_reserved:
            start_gb = start_reserved / (1024**3)
            print(f"Start reserved memory: {start_gb:.2f} GB")

    # Save training stats to json for GitHub reproducibility
    try:
        out_stats = {
            "args": vars(args),
            "train_result": stats.metrics if hasattr(stats, "metrics") else {},
            "trl_sft_config": asdict(sft_cfg),
        }
        with open(os.path.join(args.output_dir, "train_stats.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(out_stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    # Save the LoRA adapters
    model.save_pretrained("Qwen3_Lora")
    # Save locally to 16bit
    model.save_pretrained_merged("Qwen3_Lora_float16")


if __name__ == "__main__":
    main()
