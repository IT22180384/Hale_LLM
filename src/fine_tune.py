#!/usr/bin/env python3
"""
Fine-tuning script for LLaMA 3 with LoRA/QLoRA support.

This script uses Hugging Face's transformers and peft libraries for efficient fine-tuning.
The original Meta model code uses fairscale for distributed training, but for LoRA fine-tuning,
we recommend using the Hugging Face format for better compatibility with peft.

Usage:
    python src/fine_tune.py \
        --model_path models/llama-3-8B \
        --data_path data/dementia_dialogues.jsonl \
        --output_dir outputs/lora_adapter \
        --use_qlora

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_jsonl_data(data_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_conversation(sample: Dict) -> str:
    """
    Format a conversation sample into a string for fine-tuning.

    Expected format in JSONL:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    or
    {"instruction": "...", "input": "...", "output": "..."}
    or
    {"prompt": "...", "response": "..."}
    """
    # Handle different data formats
    if "messages" in sample:
        # Chat format
        text = ""
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        return f"<|begin_of_text|>{text}"

    elif "instruction" in sample:
        # Alpaca format
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        output = sample["output"]

        if input_text:
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nInput: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        return prompt

    elif "prompt" in sample:
        # Simple prompt-response format
        prompt = sample["prompt"]
        response = sample["response"]
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"

    else:
        raise ValueError(f"Unknown data format: {sample.keys()}")


def create_dataset(data_path: str, tokenizer, max_length: int = 512) -> Dataset:
    """Create a Hugging Face dataset from JSONL data."""
    raw_data = load_jsonl_data(data_path)

    if not raw_data:
        raise ValueError(f"No data found in {data_path}")

    formatted_data = []
    for sample in raw_data:
        try:
            text = format_conversation(sample)
            formatted_data.append({"text": text})
        except Exception as e:
            print(f"Warning: Skipping sample due to error: {e}")
            continue

    if not formatted_data:
        raise ValueError("No valid samples found after formatting")

    dataset = Dataset.from_list(formatted_data)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    return tokenized_dataset


def setup_model_and_tokenizer(
    model_path: str,
    use_qlora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """
    Load model and tokenizer with optional QLoRA quantization.

    Note: This expects Hugging Face format models. If you have Meta's original format,
    convert it first using: python -m transformers.models.llama.convert_llama_weights_to_hf
    """
    # Default target modules for LLaMA
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config for QLoRA
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # For Mac/CPU training, use CPU explicitly to avoid MPS issues
        device = "cpu" if not torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    use_qlora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
    fp16: bool = False,
    bf16: bool = True,
):
    """
    Fine-tune LLaMA model with LoRA/QLoRA.
    """
    print(f"Loading model from {model_path}")
    model, tokenizer = setup_model_and_tokenizer(
        model_path=model_path,
        use_qlora=use_qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    print(f"Loading dataset from {data_path}")
    train_dataset = create_dataset(data_path, tokenizer, max_length=max_length)

    print(f"Dataset size: {len(train_dataset)} samples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=fp16,
        bf16=bf16,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with LoRA/QLoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (HuggingFace format)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for adapter")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16")

    args = parser.parse_args()

    train(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
    )


if __name__ == "__main__":
    main()
