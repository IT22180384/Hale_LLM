"""
Llama 3.1 Inference Wrapper
Supports Meta's native format for text generation
"""

import torch
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add llama models to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from llama3.generation import Llama
except ImportError:
    # Fallback: use llama_models package
    try:
        from llama_models.llama3.api.chat_format import ChatFormat
        from llama_models.llama3.api.tokenizer import Tokenizer
        from llama_models.llama3.reference_impl.generation import Llama
    except ImportError:
        print("Error: Could not import Llama modules")
        print("Please ensure llama3 implementation is in src/llama3/")
        raise


class LlamaInference:
    """Wrapper for Llama 3.1 inference"""

    def __init__(
        self,
        ckpt_dir: str,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        model_parallel_size: Optional[int] = None,
    ):
        """
        Initialize Llama model

        Args:
            ckpt_dir: Path to checkpoint directory
            tokenizer_path: Path to tokenizer model (optional, will use ckpt_dir/tokenizer.model if not provided)
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
            model_parallel_size: Model parallel size (None for auto-detect)
        """
        self.ckpt_dir = Path(ckpt_dir)

        if tokenizer_path is None:
            tokenizer_path = str(self.ckpt_dir / "tokenizer.model")

        print(f"Loading model from {ckpt_dir}...")

        self.generator = Llama.build(
            ckpt_dir=str(self.ckpt_dir),
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )

        print("✓ Model loaded successfully")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_gen_len: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response from chat messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_gen_len: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated response text
        """
        # Format for chat completion
        dialogs = [messages]

        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract response
        if results and len(results) > 0:
            return results[0]['generation']['content']
        else:
            return ""

    def text_completion(
        self,
        prompts: List[str],
        max_gen_len: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text completion (non-chat format)

        Args:
            prompts: List of text prompts
            max_gen_len: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of generated texts
        """
        results = self.generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        return [r['generation'] for r in results]


if __name__ == "__main__":
    # Test the inference
    ckpt_dir = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B-Instruct"

    if not ckpt_dir.exists():
        print(f"Error: Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    print("Testing Llama inference...")

    llama = LlamaInference(
        ckpt_dir=str(ckpt_dir),
        max_seq_len=2048,
        max_batch_size=1
    )

    # Test chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you today?"}
    ]

    print("\nGenerating response...")
    response = llama.generate(messages, max_gen_len=100)
    print(f"\nResponse: {response}")
