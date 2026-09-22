#!/usr/bin/env python3
"""
Mac Lab — Apple Silicon SFT/LoRA Training

A self-contained training script for LoRA fine-tuning on Apple Silicon Macs
using PyTorch MPS. This is NOT part of the Halo CUDA training stack.

Usage:
    python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml
    python examples/lab-mac/train_sft_lora.py config.yaml --max_steps=100 --learning_rate=1e-4
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


@dataclass
class LabMacConfig:
    """Configuration for Mac Lab training."""

    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft[:1000]"
    conversation_field: str = "messages"

    max_length: int = 2048
    max_steps: int = 20
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    gradient_checkpointing: bool = True
    output_dir: str = ""
    logging_steps: int = 1
    save_steps: int = 0
    save_total_limit: int = 2
    seed: int = 42

    report_to: str = "none"
    allow_cpu_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = Path(self.model_name_or_path).name.lower().replace(".", "-")
            self.output_dir = f"checkpoints/lab-mac-{model_short}-{timestamp}"
        if self.save_steps == 0:
            self.save_steps = max(1, self.max_steps // 2)


def verify_macos_mps(allow_cpu: bool = False) -> torch.device:
    """
    Verify MPS is available and return the device.

    Exits with a clear error message if MPS is not available, unless
    CPU fallback is explicitly allowed (for testing only).
    """
    print("=" * 60)
    print("Mac Lab — Apple Silicon SFT/LoRA Training")
    print("=" * 60)
    print()
    print("NOTE: This is NOT White Circle Halo CUDA training.")
    print("      This is a fork-local lab for Apple Silicon experimentation.")
    print()

    if not hasattr(torch.backends, "mps"):
        if allow_cpu:
            print("WARNING: MPS not available (old PyTorch?). Using CPU (SLOW).")
            return torch.device("cpu")
        print("ERROR: PyTorch MPS backend not found.")
        print("       Ensure you have PyTorch 2.0+ installed.")
        print("       pip install torch>=2.3.0")
        sys.exit(1)

    if not torch.backends.mps.is_available():
        if allow_cpu:
            print("WARNING: MPS not available. Using CPU (SLOW).")
            return torch.device("cpu")
        print("ERROR: MPS is not available on this system.")
        print()
        print("Possible causes:")
        print("  - Not running on Apple Silicon (M1/M2/M3/M4/M5)")
        print("  - macOS version too old (need 13.0+)")
        print()
        print("Check: python -c \"import torch; print(torch.backends.mps.is_available())\"")
        sys.exit(1)

    if not torch.backends.mps.is_built():
        if allow_cpu:
            print("WARNING: MPS not built into this PyTorch. Using CPU (SLOW).")
            return torch.device("cpu")
        print("ERROR: PyTorch was not built with MPS support.")
        print("       Reinstall PyTorch: pip install --force-reinstall torch>=2.3.0")
        sys.exit(1)

    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    return torch.device("mps")


def load_config(config_path: str, cli_overrides: dict[str, Any]) -> LabMacConfig:
    """Load config from YAML and apply CLI overrides."""
    config_dict: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

    config_dict.update(cli_overrides)

    for key in ["lora_target_modules"]:
        if key in config_dict and isinstance(config_dict[key], str):
            config_dict[key] = [m.strip() for m in config_dict[key].split(",")]

    return LabMacConfig(**config_dict)


def format_chat_messages(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
    conversation_field: str,
    max_length: int,
) -> dict[str, Any]:
    """Format chat messages using the tokenizer's chat template."""
    messages = example.get(conversation_field, [])
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mac Lab — Apple Silicon SFT/LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml
    python examples/lab-mac/train_sft_lora.py config.yaml --max_steps=50 --learning_rate=5e-5
        """,
    )
    parser.add_argument("config", nargs="?", default="", help="Path to YAML config file")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--dataset_name", type=str, help="Dataset name")
    parser.add_argument("--dataset_split", type=str, help="Dataset split")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--report_to", type=str, help="Reporting backend (none, wandb)")
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="Allow CPU fallback if MPS unavailable (SLOW)",
    )

    args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config = load_config(args.config, cli_overrides)

    device = verify_macos_mps(allow_cpu=config.allow_cpu_fallback)

    print(f"Model: {config.model_name_or_path}")
    print(f"Dataset: {config.dataset_name} ({config.dataset_split})")
    print(f"Max steps: {config.max_steps}")
    print(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"Output: {config.output_dir}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = model.to(device)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    print("Loading dataset...")
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
    )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda ex: format_chat_messages(ex, tokenizer, config.conversation_field, config.max_length),
        remove_columns=dataset.column_names,
        num_proc=1,
    )
    tokenized_dataset = tokenized_dataset.filter(lambda ex: len(ex["input_ids"]) > 0)

    print(f"Dataset size: {len(tokenized_dataset)} examples")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        use_cpu=device.type == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    trainer.train()

    print()
    print("Saving final checkpoint...")
    final_path = Path(config.output_dir) / "checkpoint-final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print(f"Checkpoints saved to: {config.output_dir}")
    print(f"Final adapter: {final_path}")
    print()
    print("To merge the adapter into the base model:")
    print()
    print("    from peft import PeftModel")
    print("    from transformers import AutoModelForCausalLM")
    print()
    print(f'    base = AutoModelForCausalLM.from_pretrained("{config.model_name_or_path}")')
    print(f'    model = PeftModel.from_pretrained(base, "{final_path}")')
    print("    merged = model.merge_and_unload()")
    print('    merged.save_pretrained("my-merged-model")')
    print()


if __name__ == "__main__":
    main()
