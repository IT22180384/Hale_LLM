# Hale_LLM

A fine-tuned LLaMA 3 8B model for dementia care dialogues.

## Project Structure

```
Hale_LLM/
├── data/
│   └── dementia_dialogues.jsonl    # Training dataset
├── models/
│   └── llama-3-8B/                 # Model weights (download separately)
│       └── README.md               # Download instructions
├── scripts/
│   └── chat_completion.py          # Inference script
├── src/
│   ├── llama3/                     # LLaMA 3 model implementation
│   │   ├── __init__.py
│   │   ├── args.py                 # Model configuration
│   │   ├── model.py                # Transformer architecture
│   │   ├── generation.py           # Generation/inference
│   │   ├── tokenizer.py            # Tokenizer
│   │   ├── tokenizer.model         # Tokenizer binary
│   │   ├── chat_format.py          # Chat formatting
│   │   └── tool_utils.py           # Tool utilities
│   ├── checkpoint.py               # Checkpoint handling
│   ├── datatypes.py                # Type definitions
│   ├── tokenizer_utils.py          # Tokenizer utilities
│   ├── fine_tune.py                # LoRA/QLoRA fine-tuning
│   ├── inference.py                # Inference module
│   └── websocket_server.py         # WebSocket server
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

You cannot redistribute LLaMA weights. Follow these steps to download:

#### Option A: Using Meta's Official CLI (Recommended for original format)

1. Request access at: https://llama.meta.com/llama-downloads/
2. Once approved:
   ```bash
   pip install llama-models
   llama download --model-id Llama-3.1-8B-Instruct
   ```
3. Move files to `models/llama-3-8B/`

#### Option B: Using Hugging Face (Recommended for fine-tuning)

1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Download:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/llama-3-8B
   ```

See `models/llama-3-8B/README.md` for detailed instructions.

## Usage

### Inference with Meta's Original Format

For distributed inference using Meta's original implementation:

```bash
torchrun --nproc_per_node=1 scripts/chat_completion.py \
    --ckpt_dir models/llama-3-8B \
    --max_seq_len 512 \
    --max_batch_size 4
```

### Fine-tuning with LoRA/QLoRA

Fine-tune on your dataset using the HuggingFace-compatible script:

```bash
# Standard LoRA
python src/fine_tune.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/dementia_dialogues.jsonl \
    --output_dir outputs/lora_adapter \
    --num_epochs 3 \
    --batch_size 4

# QLoRA (4-bit quantization - less memory)
python src/fine_tune.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/dementia_dialogues.jsonl \
    --output_dir outputs/qlora_adapter \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 4
```

### Fine-tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha scaling |
| `--lora_dropout` | 0.05 | Dropout probability |
| `--num_epochs` | 3 | Training epochs |
| `--batch_size` | 4 | Batch size per device |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation |
| `--learning_rate` | 2e-4 | Learning rate |
| `--max_length` | 512 | Max sequence length |

## Data Format

The fine-tuning script supports multiple data formats in JSONL:

### Chat Format
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Alpaca Format
```json
{"instruction": "...", "input": "...", "output": "..."}
```

### Simple Format
```json
{"prompt": "...", "response": "..."}
```

## Hardware Requirements

- **Inference**: 16GB+ VRAM for full precision, 8GB+ for quantized
- **LoRA Fine-tuning**: 16GB+ VRAM
- **QLoRA Fine-tuning**: 8GB+ VRAM (4-bit quantization)

## License

- **Code**: Apache 2.0
- **LLaMA Model**: Subject to Meta's LLaMA Community License Agreement

## Acknowledgments

Model architecture and original code from [Meta's llama-models](https://github.com/meta-llama/llama-models).
