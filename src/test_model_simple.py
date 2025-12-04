#!/usr/bin/env python3
"""
Simple model test - debug script
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "models" / "phi_pitt_lora_final"

print("=" * 70)
print("🔧 Simple Model Test - Debug")
print("=" * 70)
print()

# Step 1: Load base model
print("Step 1: Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print("✅ Base model loaded\n")

# Step 2: Load tokenizer
print("Step 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded\n")

# Step 3: Load LoRA adapter
print("Step 3: Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
model.eval()
print("✅ LoRA adapter loaded\n")

# Step 4: Simple test generation
print("=" * 70)
print("Step 4: Testing generation...")
print("=" * 70)
print()

test_prompt = "Hello, how are you?"

print(f"Input prompt: {test_prompt}\n")

# Format like training data
formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{test_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

print(f"Formatted prompt:\n{formatted_prompt}\n")

# Tokenize
print("Tokenizing...")
input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
print(f"Token count: {input_ids.shape[1]}\n")

# Generate
print("Generating response (this may take 30-60 seconds on Mac)...")
print("-" * 70)

try:
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("✅ Generation completed!\n")

    # Decode
    print("Decoding output...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("-" * 70)
    print()
    print(f"Full response:\n{response}")
    print()
    print("-" * 70)

except Exception as e:
    print(f"❌ Error during generation: {e}")
    import traceback
    traceback.print_exc()
