# PaliGemma Inference Pipeline

Replication and efficient inference for the PaliGemma model—a state-of-the-art vision-language model combining a SigLIP vision encoder with a Gemma language decoder. This pipeline has been optimized for both CPU and Apple Silicon (MPS) devices with advanced techniques such as flash attention, paged KV caching, speculative decoding, and multi-token prediction.

## Features

- **Automatic Device Selection:**  
  Automatically selects the best available device (MPS for Apple Silicon or CPU) with fallback mechanisms.

- **Advanced Device-Specific Optimizations:**  
  - **MPS (Apple Silicon) Optimizations:**
    - Automatic mixed precision (float16) and half‑precision conversion.
    - Metal-specific memory layout optimizations.
    - Optimized memory formats for Metal Performance Shaders.
    - Built-in caching optimizations.
    - Automatic fallback to CPU if MPS is unavailable.
  - **CPU Optimizations:**
    - Dynamic quantization of linear layers (int8) for faster inference.
    - Optimized memory layouts (e.g., channels_last when beneficial).
    - CPU-specific automatic mixed precision and inference mode enhancements.

- **Advanced Attention Mechanisms:**  
  - **Flash Attention with Paged KV Cache:**  
    - A new `GemmaFlashPagedAttention` module leveraging PyTorch’s efficient scaled dot‑product attention (flash attention–like kernels) with support for rotary embeddings and paged KV caching.
  
- **Accelerated Decoding:**  
  - **Speculative Decoding & Multi‑Token Prediction:**  
    - A parallelized prefill stage that processes the prompt once.
    - Block token generation (predicting multiple tokens per iteration using the `n_predict` parameter).
    - Top‑p nucleus sampling with temperature scaling for faster and more efficient decoding.

- **Multimodal Support:**  
  Supports both image and text inputs with a unified processing pipeline.

- **Customizable Inference Parameters:**  
  Easily configure prompt text, image inputs, maximum tokens, sampling temperature, top‑p threshold, and number of tokens predicted in parallel.

- **Efficient Memory Management:**  
  Employs proper context handling and caching (KV cache) to manage memory during inference.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/codingwithsurya/PaliGemma-Inference-Pipeline.git
   cd paligemma-inference
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token:
   - Copy `.env.template` to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace `your_token_here` with your Hugging Face token
   - You can get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

## Usage

Run inference using the following command:
```bash
# For MPS (Apple Silicon GPU) - Recommended for Mac users
python inference.py --prompt "Describe this image" --image_file_path "path/to/your/image.jpg"

# For CPU-only inference
python inference.py --prompt "Describe this image" --image_file_path "path/to/your/image.jpg" --only_cpu

# Using a sample image with accelerated decoding (block prediction)
python inference.py --prompt "Describe this image in detail" --image_file_path dog.jpg --max_tokens_to_generate 300 --n_predict 4

```

### Parameters

- `--prompt`: The text prompt for the model.
- `--image_file_path`: Path to the input image.
- `--only_cpu`: Flag to force CPU-only inference (default: False; will use MPS if available).
- `--max_tokens_to_generate`: Maximum number of tokens to generate (default: 300).
- `--temperature`: Sampling temperature (default: 0.7).
- `--top_p`: Top‑p sampling parameter (default: 0.9).
- `--n_predict`: Number of tokens to predict in parallel during accelerated decoding (default: 4).

## Technical Details

This project leverages several advanced deep learning concepts and optimizations:

- **Architecture:**
  - **Vision Encoder:** SigLIP vision encoder processes image inputs.
  - **Language Decoder:** Gemma language decoder based on the Transformer architecture with rotary positional embeddings.
  - **Attention Mechanism:**  
    The newly introduced `GemmaFlashPagedAttention` module uses flash attention kernels and supports paged KV caching for efficient multi-head attention.

- **Optimizations:**
  - **Device-Specific Enhancements:**
    - **CPU:** Dynamic quantization (int8) for linear layers, optimized memory layouts, and CPU-specific inference mode.
    - **MPS:** Half-precision conversion, automatic mixed precision, and Metal-specific memory optimizations.
  - **Accelerated Decoding:**
    - Speculative decoding and multi-token prediction allow for block token generation, reducing latency.
    - Parallel prefill stage for efficient prompt processing.
    - Top‑p sampling with temperature scaling ensures quality output while speeding up inference.
  
- **Memory Management:**
  - Efficient tensor operations with proper context management.
  - KV caching for fast sequential decoding.
  - Automatic device selection with fallback support.

## Performance

- **Apple Silicon (MPS):**  
  Uses MPS (Metal Performance Shaders) with half‑precision to accelerate inference.
  
- **CPU:**  
  Optimized for CPU inference using int8 dynamic quantization and efficient memory layouts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Umar Jamil's VLM Tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw)
- [PaLiGemma Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/paligemma)
- [PyTorch](https://pytorch.org/)
```
