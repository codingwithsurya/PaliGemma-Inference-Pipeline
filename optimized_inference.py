import torch
import torch.nn as nn
from typing import Optional, Dict
import math
from contextlib import contextmanager
from PIL import Image
from gemma_model import KVCache
import numpy as np

# -------------------------
# New InferenceOptimizer:
# Applies targeted quantization:
#   - CPU: dynamic quantization to int8 (for nn.Linear modules)
#   - MPS: convert model to half precision.
# Also provides an inference context manager.
# -------------------------
class InferenceOptimizer:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.is_mps = (device == "mps")
        self.is_cpu = (device == "cpu")
    
    def optimize_model(self):
        if self.is_cpu:
            # Apply dynamic quantization for int8 inference on CPU
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            self.model.to(self.device)
        elif self.is_mps:
            # For MPS, use half precision
            self.model = self.model.half()
            self.model.to(self.device)
        else:
            self.model.to(self.device)
        return self.model
    
    @contextmanager
    def inference_context(self):
        # Use torch.inference_mode and autocast (with appropriate dtype)
        if self.is_mps:
            with torch.inference_mode(), torch.autocast(device_type='mps', dtype=torch.float16), torch.no_grad():
                yield
        else:
            with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float32), torch.no_grad():
                yield

# -------------------------
# OptimizedInference wrapper:
#   - Prepares inputs via the processor.
#   - Implements an accelerated generate() method that pre-fills the prompt once then decodes tokens in blocks.
#   - Simulates speculative decoding / multi-token prediction.
# -------------------------
class OptimizedInference:
    def __init__(self, model, processor, device: str):
        self.device = device
        self.optimizer = InferenceOptimizer(model, device)
        self.model = self.optimizer.optimize_model()
        self.processor = processor
        self.setup_caching()

    def setup_caching(self):
        # Initialize caching dictionaries (if needed for more advanced KV cache management)
        self.kv_cache = KVCache()

    def prepare_inputs(self, image_path: str, prompt: str) -> Dict[str, torch.Tensor]:
        image = Image.open(image_path)
        inputs = self.processor(text=[prompt], images=[image])
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Sample a token from probabilities using top-p (nucleus) filtering.
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the mask to keep at least one token.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs = probs.clone()
        probs[sorted_indices_to_remove] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True) + 1e-8)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n_predict: int = 4
    ):
        """
        Accelerated text generation with:
          - Prefill stage: process prompt in one forward pass.
          - Block generation: predict n_predict tokens at a time using speculative decoding / multi-token prediction.
        """
        inputs = self.prepare_inputs(image_path, prompt)
        input_ids = inputs["input_ids"]  # shape: [batch, seq_len]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        # Use the prefill phase: run the prompt once
        with self.optimizer.inference_context():
            _ = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=self.kv_cache
            )

        # Initialize generated tokens with the prompt.
        generated_tokens = input_ids
        stop_token = self.processor.tokenizer.eos_token_id

        current_length = generated_tokens.shape[1]
        # Loop until maximum length is reached.
        while current_length < max_length:
            with self.optimizer.inference_context():
                # Prepare a block: append n_predict dummy tokens (e.g., zeros) to the current tokens.
                dummy_tokens = torch.zeros((generated_tokens.shape[0], n_predict), dtype=generated_tokens.dtype, device=generated_tokens.device)
                block_input_ids = torch.cat([generated_tokens, dummy_tokens], dim=1)
                block_attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], n_predict), device=attention_mask.device, dtype=attention_mask.dtype)],
                    dim=1
                )
                outputs = self.model(
                    input_ids=block_input_ids,
                    pixel_values=pixel_values,
                    attention_mask=block_attention_mask,
                    kv_cache=self.kv_cache
                )
                logits = outputs["logits"]
                # Extract logits corresponding to the new (dummy) positions.
                block_logits = logits[:, -n_predict:, :]  # [B, n_predict, vocab_size]
                next_tokens = []
                for i in range(n_predict):
                    token_logits = block_logits[:, i, :] / temperature
                    token_probs = torch.softmax(token_logits, dim=-1)
                    next_token = self._sample_top_p(token_probs, top_p)
                    next_tokens.append(next_token)
                next_tokens = torch.cat(next_tokens, dim=1)  # [B, n_predict]

            # Check if any token is the EOS token; if so, truncate the block.
            eos_mask = (next_tokens == stop_token)
            if eos_mask.any():
                # Find first occurrence of EOS (for batch index 0 as representative)
                eos_pos = (next_tokens[0] == stop_token).nonzero(as_tuple=False)[0].item()
                next_tokens = next_tokens[:, :eos_pos+1]
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
                break
            else:
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
            
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], n_predict), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1
            )
            current_length = generated_tokens.shape[1]

        # Decode generated tokens (assumes batch size of 1)
        decoded = self.processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return decoded
