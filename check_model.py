#!/usr/bin/env python3
"""
Quick script to check if Meta's Llama model is properly installed
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

MODEL_PATH = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B-Instruct"

print("=" * 70)
print("Checking Llama 3.1 Model Installation")
print("=" * 70)
print()

print(f"Expected model location: {MODEL_PATH}")
print()

# Check if directory exists
if not MODEL_PATH.exists():
    print("❌ Model directory NOT found!")
    print(f"   Please ensure model is downloaded to: {MODEL_PATH}")
    sys.exit(1)

print("✓ Model directory exists")
print()

# Check for Meta's official format files
required_files = {
    "consolidated.00.pth": "Model weights (Meta format)",
    "params.json": "Model parameters",
    "tokenizer.model": "Tokenizer"
}

print("Checking for Meta's official format files:")
print("-" * 70)

meta_format_detected = True
for filename, description in required_files.items():
    filepath = MODEL_PATH / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ {filename:25s} ({description})")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"❌ {filename:25s} - NOT FOUND")
        meta_format_detected = False

print()

if meta_format_detected:
    print("=" * 70)
    print("✓ Meta's official format detected!")
    print("=" * 70)
    print()
    print("The chatbot will use Meta's official Llama format.")
    print()

    # Test the wrapper
    print("Testing Meta wrapper...")
    try:
        from meta_llama_wrapper import MetaLlamaWrapper
        print("✓ MetaLlamaWrapper imported successfully")

        wrapper = MetaLlamaWrapper(str(MODEL_PATH))
        print("✓ MetaLlamaWrapper initialized successfully")
        print()
        print("=" * 70)
        print("✓ Everything looks good! Ready to start the API server.")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error testing wrapper: {e}")
        print()
        print("You may need to install llama-models package:")
        print("  pip install llama-models")
else:
    print("=" * 70)
    print("⚠ Meta's official format NOT complete")
    print("=" * 70)
    print()
    print("The chatbot will fall back to HuggingFace format.")
    print("You'll need to:")
    print("  1. Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("  2. Login with: huggingface-cli login")

print()
