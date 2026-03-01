from __future__ import annotations

import math
import types
from typing import Callable

import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    repeat_kv,
)


def _custom_c2fab_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eager attention path with C2FAB bias injection into attention logits.
    """
    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    # Exact logits line requested for monkey-patch control.
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        module.head_dim
    )

    c2fab_bias_2d = kwargs.pop("c2fab_bias_2d", None)
    c2fab_layer_alpha = kwargs.pop("c2fab_layer_alpha", None)
    if c2fab_bias_2d is not None and c2fab_layer_alpha is not None:
        # c2fab_bias_2d:[batch_size, q_len_bias, kv_len_bias]
        q_len = attn_weights.shape[-2]
        kv_len = attn_weights.shape[-1]
        if c2fab_bias_2d.ndim != 3:
            raise ValueError(
                f"c2fab_bias_2d must be rank-3 [batch,q,kv], got shape {tuple(c2fab_bias_2d.shape)}."
            )

        if c2fab_bias_2d.shape[-2] != q_len:
            if c2fab_bias_2d.shape[-2] > q_len:
                c2fab_bias_2d = c2fab_bias_2d[:, -q_len:, :]
            else:
                q_pad = torch.zeros(
                    (
                        c2fab_bias_2d.shape[0],
                        q_len - c2fab_bias_2d.shape[-2],
                        c2fab_bias_2d.shape[-1],
                    ),
                    dtype=c2fab_bias_2d.dtype,
                    device=c2fab_bias_2d.device,
                )
                c2fab_bias_2d = torch.cat([q_pad, c2fab_bias_2d], dim=-2)

        if c2fab_bias_2d.shape[-1] != kv_len:
            if c2fab_bias_2d.shape[-1] > kv_len:
                c2fab_bias_2d = c2fab_bias_2d[..., -kv_len:]
            else:
                k_pad = torch.zeros(
                    (
                        c2fab_bias_2d.shape[0],
                        c2fab_bias_2d.shape[1],
                        kv_len - c2fab_bias_2d.shape[-1],
                    ),
                    dtype=c2fab_bias_2d.dtype,
                    device=c2fab_bias_2d.device,
                )
                c2fab_bias_2d = torch.cat([k_pad, c2fab_bias_2d], dim=-1)

        if c2fab_layer_alpha.numel() != attn_weights.shape[1]:
            raise ValueError(
                "Alpha head count mismatch: "
                f"expected {attn_weights.shape[1]}, got {c2fab_layer_alpha.numel()}."
            )

        bias = c2fab_bias_2d.unsqueeze(1)  # [batch_size, 1, q_len, kv_len]
        head_alpha = c2fab_layer_alpha.view(1, -1, 1, 1)  # [1, num_q_heads, 1, 1]
        bias = (bias * head_alpha).to(dtype=attn_weights.dtype, device=attn_weights.device)
        # Keep patched logits in a numerically safe range.
        bias = torch.clamp(bias, min=-8.0, max=8.0)
        attn_weights = attn_weights + bias

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def custom_c2fab_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Mirrors the current Hugging Face attention forward structure and injects C2FAB bias.
    """
    if not hasattr(self, "c2fab_heads") or self.c2fab_heads is None:
        raise AttributeError("self.c2fab_heads is missing. Call apply_c2fab_patch first.")
    if not hasattr(self, "layer_idx") or self.layer_idx is None:
        raise AttributeError("self.layer_idx is missing. Call apply_c2fab_patch first.")
    if self.layer_idx < 32 or self.layer_idx > 35:
        raise ValueError(
            f"C2FAB patch only supports layers 32-35, got layer_idx={self.layer_idx}."
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    heads_param = next(self.c2fab_heads.parameters())
    heads_input = hidden_states.to(device=heads_param.device, dtype=heads_param.dtype)
    Phi_u, R_q, _ = self.c2fab_heads(heads_input, heads_input, use_bidirectional=True)
    raw_bias = torch.einsum("bqd,bud->bqu", R_q, Phi_u)  # raw_bias:[batch_size, q_len, kv_len]
    layer_alpha = self.c2fab_heads.alphas[self.layer_idx - 32]  # [num_q_heads]

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, _custom_c2fab_eager_attention_forward
    )
    attention_interface = _custom_c2fab_eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        c2fab_bias_2d=raw_bias,
        c2fab_layer_alpha=layer_alpha,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_c2fab_patch(
    model,
    trained_heads_module,
    target_layers: list[int] = [32, 33, 34, 35],
):
    """
    Patch selected Mistral self-attention layers with custom_c2fab_forward.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a model with model.model.layers (Mistral-style structure).")
    if trained_heads_module is None:
        raise ValueError("trained_heads_module must not be None.")

    layer_count = len(model.model.layers)
    for layer_idx in target_layers:
        if layer_idx < 0 or layer_idx >= layer_count:
            raise IndexError(
                f"target layer {layer_idx} is out of range for model with {layer_count} layers."
            )

    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        model_device = model.get_input_embeddings().weight.device
    else:
        model_device = next(model.parameters()).device
    trained_heads_module = trained_heads_module.to(model_device)

    for layer_idx in target_layers:
        attn_module = model.model.layers[layer_idx].self_attn
        attn_module.c2fab_heads = trained_heads_module
        attn_module.layer_idx = layer_idx
        attn_module.forward = types.MethodType(custom_c2fab_forward, attn_module)

    return model
