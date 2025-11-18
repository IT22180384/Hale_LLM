#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""
Chat completion inference script for LLaMA 3.

Usage:
    torchrun --nproc_per_node=1 scripts/chat_completion.py \
        --ckpt_dir models/llama-3-8B \
        --max_seq_len 512 \
        --max_batch_size 4
"""

import os
import sys
from pathlib import Path
from typing import Optional

import fire
import torch
from termcolor import cprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datatypes import RawMessage, StopReason
from llama3.generation import Llama3


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
):
    """
    Run chat completion inference.

    Args:
        ckpt_dir: Path to model checkpoint directory
        temperature: Sampling temperature (0 = greedy)
        top_p: Top-p sampling parameter
        max_seq_len: Maximum sequence length
        max_batch_size: Maximum batch size
        world_size: Number of model parallel processes
        quantization_mode: Quantization mode (fp8_mixed, int4_mixed, or None)
    """
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    # Example dialogs
    dialogs = [
        [RawMessage(role="user", content="what is the recipe of mayonnaise?")],
        [
            RawMessage(
                role="user",
                content="I am going to Paris, what should I see?",
            ),
            RawMessage(
                role="assistant",
                content="""\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
                stop_reason=StopReason.end_of_turn,
            ),
            RawMessage(role="user", content="What is so great about #1?"),
        ],
        [
            RawMessage(role="system", content="Always answer with Haiku"),
            RawMessage(role="user", content="I am going to Paris, what should I see?"),
        ],
        [
            RawMessage(role="system", content="Always answer with emojis"),
            RawMessage(role="user", content="How to go from Beijing to NY?"),
        ],
    ]

    for dialog in dialogs:
        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        batch = [dialog]
        for token_results in generator.chat_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
        ):
            result = token_results[0]
            if result.finished:
                break

            cprint(result.text, color="yellow", end="")
        print("\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
