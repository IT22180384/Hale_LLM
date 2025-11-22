"""
Simple wrapper for Meta's Llama format
Uses torchrun and Meta's official scripts
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import tempfile


class MetaLlamaWrapper:
    """Wrapper for Meta's official Llama format"""

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = Path(ckpt_dir)
        self.project_root = Path(__file__).parent.parent

        # Verify checkpoint exists
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        print(f"✓ Using Meta's Llama format from: {ckpt_dir}")

    def generate(self, messages: List[Dict], max_gen_len: int = 256, temperature: float = 0.7) -> str:
        """
        Generate response using Meta's official format

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_gen_len: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        # Create temporary file for dialog
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            dialog_data = {
                "dialogs": [messages],
                "max_gen_len": max_gen_len,
                "temperature": temperature
            }
            json.dump(dialog_data, f)
            dialog_file = f.name

        try:
            # Use Meta's chat completion script
            script_path = self.project_root / "scripts" / "meta_inference.py"

            # Run inference
            cmd = [
                sys.executable,
                str(script_path),
                "--ckpt_dir", str(self.ckpt_dir),
                "--dialog_file", dialog_file
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"Error running inference: {result.stderr}")
                return "I apologize, but I'm having trouble generating a response."

            # Parse output
            output = json.loads(result.stdout)
            return output.get("response", "")

        except Exception as e:
            print(f"Error: {e}")
            return "I apologize, but I'm having trouble generating a response."

        finally:
            # Cleanup temp file
            try:
                Path(dialog_file).unlink()
            except:
                pass


if __name__ == "__main__":
    # Test
    ckpt_dir = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B-Instruct"

    wrapper = MetaLlamaWrapper(str(ckpt_dir))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]

    response = wrapper.generate(messages, max_gen_len=100)
    print(f"Response: {response}")
