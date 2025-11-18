# LLaMA 3 8B Instruct Model Weights

## Download Instructions

You cannot redistribute LLaMA weights. Follow these steps to download them:

### Option 1: Using Meta's Official CLI (Recommended)

1. Request access at: https://llama.meta.com/llama-downloads/
2. Once approved, install the llama-models package:
   ```bash
   pip install llama-models
   ```
3. Download the model:
   ```bash
   llama download --model-id Llama-3.1-8B-Instruct
   ```
4. Move the downloaded files to this directory

### Option 2: Using Hugging Face

1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Once approved, download using the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir .
   ```

### Expected Files

After downloading, this directory should contain:
- `consolidated.00.pth` (or multiple shards for larger models)
- `params.json`
- `tokenizer.model`

## Important Notes

- Model weights are approximately 16GB for the 8B model
- Ensure you have sufficient disk space and bandwidth
- Keep weights local - do not commit to version control
- For fine-tuning with LoRA, you may want to use the Hugging Face format

## License

LLaMA models are subject to Meta's LLaMA Community License Agreement.
See: https://github.com/meta-llama/llama-models/blob/main/LICENSE
