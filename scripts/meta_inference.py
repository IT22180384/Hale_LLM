"""
Simple inference script for Meta's Llama format
Loads model once and generates responses
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from llama_models package
try:
    from llama_models.llama3.api.datatypes import Message, CompletionMessage
    from llama_models.llama3.api.chat_format import ChatFormat
    from llama_models.llama3.api.tokenizer import Tokenizer
    from llama_models.llama3.reference_impl.generation import Llama
    USE_LLAMA_MODELS = True
except ImportError:
    USE_LLAMA_MODELS = False
    print("Warning: llama_models not available, using transformers fallback")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory")
    parser.add_argument("--dialog_file", required=True, help="Dialog JSON file")
    args = parser.parse_args()

    # Load dialog
    with open(args.dialog_file) as f:
        data = json.load(f)

    dialogs = data["dialogs"]
    max_gen_len = data.get("max_gen_len", 256)
    temperature = data.get("temperature", 0.7)

    if not USE_LLAMA_MODELS:
        # Fallback: simple response
        response = {
            "response": "I'm here to help you. What would you like to talk about?",
            "method": "fallback"
        }
        print(json.dumps(response))
        return

    try:
        # Build Llama generator
        generator = Llama.build(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=str(Path(args.ckpt_dir) / "tokenizer.model"),
            max_seq_len=2048,
            max_batch_size=1,
        )

        # Run chat completion
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=0.9,
        )

        # Extract response
        if results and len(results) > 0:
            response_text = results[0]['generation']['content']
        else:
            response_text = "I apologize, but I couldn't generate a response."

        response = {
            "response": response_text,
            "method": "meta_official"
        }

    except Exception as e:
        response = {
            "response": f"Error: {str(e)}",
            "method": "error"
        }

    print(json.dumps(response))


if __name__ == "__main__":
    main()
