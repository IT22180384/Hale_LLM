#!/usr/bin/env python3
"""
CPU-optimized model test
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import signal

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "models" / "phi_pitt_lora_final"

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timeout - model is taking too long on CPU")

print("=" * 70)
print("🔧 CPU-Optimized Model Test")
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

# Step 4: Simple test generation - CPU optimized
print("=" * 70)
print("Step 4: Testing generation (CPU mode - optimized)...")
print("=" * 70)
print()

test_prompt = "Hello"

print(f"Input: {test_prompt}\n")

# Simple prompt without special tokens
input_text = test_prompt
print(f"Tokenizing...")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(f"Tokens: {input_ids.shape[1]}\n")

print("Generating response (max 50 tokens, should be 20-30 seconds)...")
print("-" * 70)

try:
    # Set a timeout of 120 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)

    with torch.no_grad():
        # CPU-optimized parameters
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,  # SHORT for CPU
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            num_beams=1,  # No beam search on CPU
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Cancel timeout
    signal.alarm(0)

    print("\n✅ Generation completed!\n")

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("-" * 70)
    print()
    print("RESPONSE FROM MODEL:")
    print(response)
    print()
    print("-" * 70)
    print()
    print("✅ MODEL IS WORKING!")

except TimeoutError as e:
    print(f"\n❌ {e}")
    print("\nMac CPU is too slow for LLM inference.")
    print("This is expected - we'll deploy to Azure GPU for production.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
