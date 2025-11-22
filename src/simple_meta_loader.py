"""
Simple Meta Llama loader without torchrun
Initializes distributed environment in single-process mode
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict

# Set environment for single process BEFORE any imports
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add Hale_LLM to path


class SimpleMetaLlama:
    """Simple loader for Meta's Llama format without torchrun"""
    
    def __init__(self, ckpt_dir: str, max_seq_len: int = 2048, max_batch_size: int = 1):
        """
        Initialize Llama model
        
        Args:
            ckpt_dir: Path to checkpoint directory
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        
        print(f"Loading model from {ckpt_dir}...")
        
        try:
            # Import after environment is set
            from src.llama3.generation import Llama3
            
            # Initialize distributed backend for single process
            if not torch.distributed.is_initialized():
                print("Initializing single-process distributed backend...")
                import tempfile
                import shutil
                
                # Create temporary directory for file store
                temp_dir = Path(tempfile.gettempdir()) / "torch_dist"
                temp_dir.mkdir(exist_ok=True)
                
                # Clean up old files
                for f in temp_dir.glob("*"):
                    try:
                        f.unlink()
                    except:
                        pass
                
                init_file = temp_dir / "init_file"
                
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method=f"file:///{str(init_file).replace(chr(92), '/')}",  # Forward slashes for file URI
                    rank=0,
                    world_size=1
                )
            
            # Build model (this will handle tokenizer initialization)
            print("Building Llama model...")
            self.generator = Llama3.build(
                ckpt_dir=str(self.ckpt_dir),
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                world_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            print("✓ Model loaded successfully (Meta's official format)")
            
        except Exception as e:
            print(f"Error loading Meta format: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
        try:
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
                result = results[0]
                if isinstance(result, dict) and 'generation' in result:
                    return result['generation']['content']
                elif isinstance(result, dict) and 'content' in result:
                    return result['content']
                else:
                    return str(result)
            
            return "I'm having trouble generating a response."
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I'm having trouble generating a response."


if __name__ == "__main__":
    # Test
    ckpt_dir = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B-Instruct"
    
    model = SimpleMetaLlama(str(ckpt_dir))
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]
    
    response = model.generate(messages, max_gen_len=100)
    print(f"\nResponse: {response}")
