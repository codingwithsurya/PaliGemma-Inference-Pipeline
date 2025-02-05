import torch
from torch import nn
from typing import Optional, Tuple, List
import math
from siglip_model import SiglipVisionConfig, SiglipVisionModel

# -------------------------
# KVCache definition (unchanged)
# -------------------------
class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # Initialize cache for this layer.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate new keys/values to the cache along sequence length.
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

# -------------------------
# Configuration classes
# -------------------------
class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

# -------------------------
# Norm and Rotary Embedding utilities 
# -------------------------
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# -------------------------
# MLP module 
# -------------------------
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Apply GELU activation with approximate tanh mode and compute MLP
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# -------------------------
# NEW: GemmaFlashPagedAttention
#
# Replaces the original GemmaAttention with an implementation that:
#  - Computes Q, K, V projections as before.
#  - Applies rotary embeddings.
#  - Updates the KV cache (if provided) in a paged manner.
#  - Uses torch.nn.functional.scaled_dot_product_attention (which leverages flash attention kernels)
#    with causal masking.
# -------------------------
class GemmaFlashPagedAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        # For multi-query attention: repeat keys/values to match the number of query heads.
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose to separate heads.
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings.
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache if provided.
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # For multi-query attention: repeat keys and values along the head dimension.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Permute to shape [Batch, Seq_Len, Num_Heads, Head_Dim] for scaled_dot_product_attention.
        query_states = query_states.transpose(1, 2)  # [B, q_len, num_heads, head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Adjust the attention mask if provided.
        if attention_mask is not None:
            # Expect attention_mask to be of shape [B, 1, q_len, seq_len]; squeeze the head dim.
            attn_mask = attention_mask.squeeze(1) if attention_mask.dim() == 4 else attention_mask
        else:
            attn_mask = None

        # Use scaled_dot_product_attention (flash attention style) with causal masking.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
            is_causal=True
        )
        # attn_output shape: [B, q_len, num_heads, head_dim]. Permute back.
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

# -------------------------
# GemmaDecoderLayer now uses GemmaFlashPagedAttention
# -------------------------
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Use the new flash attention module.
        self.self_attn = GemmaFlashPagedAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config): # config: GemmaConfig
        super().__init__()
        self.config = config # Store the configuration
        self.padding_idx = config.pad_token_id # Get the padding token ID from the config
        self.vocab_size = config.vocab_size # Get the vocabulary size from the config

        # Embedding layer to convert token IDs to embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)] # Create a list of GemmaDecoderLayer instances
        )
        # Layer normalization applied at the end of the decoder layers
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens # Returns the embedding layer

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None, # Mask to avoid attending to certain positions (e.g., padding tokens)
        position_ids: Optional[torch.LongTensor] = None, # Position indices for positional embeddings
        inputs_embeds: Optional[torch.FloatTensor] = None, # Pre-computed input embeddings (if input_ids are not used directly)
        kv_cache: Optional[KVCache] = None, # Key/Value cache for faster inference in autoregressive decoding
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds # Initialize hidden states with input embeddings
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype) # Calculate normalization factor
        hidden_states = hidden_states * normalizer # Apply normalization to the input embeddings

        # Pass the hidden states through each decoder layer
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states) # Apply layer normalization to the final hidden states
        return hidden_states # Return the final hidden states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config): # config: GemmaConfig
        super().__init__()
        self.config = config # Store the configuration
        self.model = GemmaModel(config) # Instantiate the base Gemma model
        self.vocab_size = config.vocab_size # Get the vocabulary size from the config
        # Linear layer to project the hidden states to logits (scores for each token in the vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens # Returns the embedding layer from the base model

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight # Tie the weights of the LM head to the embedding layer for weight sharing

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None, # Mask to avoid attending to certain positions (e.g., padding tokens)
        position_ids: Optional[torch.LongTensor] = None, # Position indices for positional embeddings
        inputs_embeds: Optional[torch.FloatTensor] = None, # Pre-computed input embeddings (if input_ids are not used directly)
        kv_cache: Optional[KVCache] = None, # Key/Value cache for faster inference in autoregressive decoding
    ) -> dict:
        outputs = self.model( # Pass inputs through the base Gemma model
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(outputs) # Project the output hidden states to logits
        logits = logits.float() # Cast logits to float for numerical stability
        return_data = {"logits": logits} # Store logits in a dictionary to be returned
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache # If kv_cache is used, also return it in the dictionary
        return return_data # Return the dictionary containing logits and optionally kv_cache


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config): # config: PaliGemmaConfig
        super().__init__()
        # Linear layer to project vision features to the desired dimension
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features) # Project the image features using the linear layer
        return hidden_states # Return the projected image features


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config): # config: PaliGemmaConfig
        super().__init__()
        self.config = config # Store the configuration
        # Vision tower to process image inputs
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # Projector to align vision features with language model embeddings
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size # Get the vocabulary size from the config

        # Instantiate the language model (GemmaForCausalLM)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 # Get padding token ID from config

    def tie_weights(self):
        return self.language_model.tie_weights() # Tie weights of the language model

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape # Get embedding dimension from image features
        batch_size, sequence_length = input_ids.shape # Get batch size and sequence length from input IDs
        dtype, device = inputs_embeds.dtype, inputs_embeds.device # Get data type and device from input embeddings
        scaled_image_features = image_features / (self.config.hidden_size**0.5) # Scale image features by normalization factor

        # Initialize an embedding tensor to store final embeddings
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Create masks to identify text tokens, image tokens and padding tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expand masks to match embedding dimension
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Fill final embedding: text embeddings for text tokens
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Scatter image features into final embedding at image token positions
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Set padding token positions to zero embedding
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        min_dtype = torch.finfo(dtype).min # Get minimum value for the data type (not used in this logic)
        q_len = inputs_embeds.shape[1] # Query length from input embeddings

        # Create causal attention mask
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            ) # Full causal mask for initial step
        else:
            assert q_len == 1 # For incremental decoding, query length should be 1
            kv_len = kv_cache.num_items() + q_len # Key/value length including cached items
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            ) # Causal mask considering kv_cache
        causal_mask = causal_mask.unsqueeze(1) # Add channel dimension

        # Generate position IDs
        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1] # Get the last cumulative sum as position IDs for incremental decoding
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0) # Ensure position IDs is 2D
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device) # Create position IDs based on attention mask

        return final_embedding, causal_mask, position_ids # Return merged embeddings, causal mask, and position IDs

    def forward(
        self,
        input_ids: torch.LongTensor = None, # Input token IDs
        pixel_values: torch.FloatTensor = None, # Pixel values for image input
        attention_mask: Optional[torch.Tensor] = None, # Attention mask
        kv_cache: Optional[KVCache] = None, # Key/Value cache
    ) -> dict:
        assert torch.all(attention_mask == 1), "The input cannot be padded" # Assert that input is not padded (attention mask should be all 1s)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # Get text embeddings for input IDs
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype)) # Process image pixels through vision tower
        image_features = self.multi_modal_projector(selected_image_feature) # Project vision features
        # Merge text embeddings with projected image features
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model( # Pass merged embeddings through the language model
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        return outputs # Return the output from the language model