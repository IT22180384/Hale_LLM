#!/usr/bin/env python3
"""
Inference script for fine-tuned LLaMA-3 with LoRA adapter
Tests the trained model locally
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# Paths
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "models" / "phi_pitt_lora_final"

def load_model():
    """Load base model + LoRA adapter (optimized for Mac)"""
    print("🤖 Loading base model...")

    # For Mac: use CPU with float32
    device = "cpu"
    torch_dtype = torch.float32

    # Load base model (no quantization on Mac)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Base model loaded on {device}")

    # Load LoRA adapter
    print(f"📦 Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    print("✅ LoRA adapter loaded")

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate response from the model"""

    # Format prompt like training data
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Tokenize
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    print("=" * 70)
    print("🎯 Fine-Tuned LLaMA-3 with LoRA - Local Test")
    print("=" * 70)
    print()

    # Load model
    model, tokenizer = load_model()
    print()

    # Test prompts
    test_prompts = [
        "Hello, how are you today?",
        "What's your name?",
        "Tell me about yourself",
    ]

    print("=" * 70)
    print("🚀 Testing Model Responses")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\n📝 User: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 Assistant: {response}")
        print("-" * 70)

    # Interactive mode
    print("\n" + "=" * 70)
    print("💬 Interactive Chat Mode (type 'quit' to exit)")
    print("=" * 70)

    while True:
        user_input = input("\n📝 You: ").strip()
        if user_input.lower() == 'quit':
            print("✅ Goodbye!")
            break
        if not user_input:
            continue

        response = generate_response(model, tokenizer, user_input)
        print(f"🤖 Assistant: {response}")


if __name__ == "__main__":
    main()
