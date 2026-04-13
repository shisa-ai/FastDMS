"""
AQUA-KV cross-layer prediction cache: predict layer L from layer L-1, quantize the residual.

SingleChunkQuantizedCacheWithPredictors is a write-once cache chunk that stores:
  - Layer 0: raw or first-layer-quantized KV
  - Layer 1+: residual = (actual - predicted), quantized with HIGGS

On read, it reconstructs: predicted + dequantize(residual).

Ported from aquakv/cache_utils.py with the same logic, cleaned imports.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .quantizers import QuantizerBase


def apply_rotary_to_keys(key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to key states."""
    _, key_states = apply_rotary_pos_emb(
        q=key_states[..., :1, :], k=key_states, cos=cos, sin=sin
    )
    return key_states


def split_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """[batch, seq, hidden] → [batch, heads, seq, head_dim]"""
    assert x.ndim == 3
    return x.reshape(*x.shape[:2], -1, head_dim).transpose(1, 2)


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    """[batch, heads, seq, head_dim] → [batch, seq, hidden]"""
    assert x.ndim == 4
    return x.transpose(1, 2).flatten(-2)


class SingleChunkQuantizedCacheWithPredictors(transformers.cache_utils.Cache):
    """
    A write-once cache chunk that uses cross-layer predictors and residual HIGGS quantization.

    For layer 0: stores KV directly (or quantized with first_layer_quantizer).
    For layer L>0: predicts K_L from K_{L-1}, quantizes residual (K_L - predicted) with HIGGS.
    Same for values, using [reconstructed_keys, V_{L-1}] as predictor input.

    :param quantizer: HIGGS quantizer for residual quantization (layers 1+)
    :param first_layer_quantizer: optional quantizer for layer 0
    :param key_predictors: dict mapping layer_idx → nn.Linear predictor for keys
    :param value_predictors: dict mapping layer_idx → nn.Linear predictor for values
    :param move_predictors_to_devices: move predictors to match input device on-the-fly
    """

    def __init__(
        self,
        *,
        quantizer: QuantizerBase,
        first_layer_quantizer: Optional[QuantizerBase] = None,
        key_predictors: Optional[Dict[int, nn.Module]] = None,
        value_predictors: Optional[Dict[int, nn.Module]] = None,
        move_predictors_to_devices: bool = True,
    ):
        super().__init__(layers=[])
        self.quantizer = quantizer
        self.first_layer_quantizer = first_layer_quantizer
        self.key_predictors = key_predictors
        self.value_predictors = value_predictors
        self.move_predictors_to_devices = move_predictors_to_devices

        self.key_states_cache: dict = {}
        self.value_states_cache: dict = {}
        self.device_map: dict = {}
        self.previous_key_reconstruction = None
        self.previous_value_reconstruction = None
        self.next_layer_idx = 0
        self.seq_length = 0
        self.cos = self.sin = None
        self.head_dim = None

    def predict_next_key_states(self) -> torch.Tensor:
        if self.key_predictors is not None:
            if self.move_predictors_to_devices:
                predictor_device = next(self.key_predictors[self.next_layer_idx].parameters()).device
                if predictor_device != self.previous_key_reconstruction.device:
                    self.key_predictors[self.next_layer_idx].to(self.previous_key_reconstruction.device)
            return self.key_predictors[self.next_layer_idx](self.previous_key_reconstruction)
        else:
            return torch.zeros_like(self.previous_key_reconstruction)

    def predict_next_value_states(self, reconstructed_key_states: torch.Tensor) -> torch.Tensor:
        if self.value_predictors is not None:
            value_predictor_inputs = torch.cat(
                [
                    reconstructed_key_states.to(self.previous_value_reconstruction.device),
                    self.previous_value_reconstruction,
                ],
                dim=-1,
            )
            if self.move_predictors_to_devices:
                predictor_device = next(self.value_predictors[self.next_layer_idx].parameters()).device
                if predictor_device != value_predictor_inputs.device:
                    self.value_predictors[self.next_layer_idx].to(value_predictor_inputs.device)
            return self.value_predictors[self.next_layer_idx](value_predictor_inputs)
        else:
            return torch.zeros_like(self.previous_value_reconstruction)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        assert layer_idx == 0
        return self.key_states_cache[0].shape[-2] if self.key_states_cache else 0

    @torch.no_grad()
    def update(
        self,
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx in (self.next_layer_idx, 0), (layer_idx, self.next_layer_idx, 0)
        assert (key_states is None and value_states is None) or (key_states.shape == value_states.shape)
        saving_new_entries = key_states is not None and key_states.numel() != 0
        assert saving_new_entries == (layer_idx not in self.key_states_cache), "can only write once per layer"
        assert key_states.device == value_states.device and key_states.dtype == value_states.dtype

        if saving_new_entries:  # write mode
            device, dtype = key_states.device, key_states.dtype
            key_states_original, value_states_original = key_states, value_states
            has_rotary_kwargs = cache_kwargs is not None and "sin" in cache_kwargs and "cos" in cache_kwargs

            if has_rotary_kwargs and self.cos is None:
                self.cos, self.sin = cache_kwargs["cos"], cache_kwargs["sin"]

            if self.head_dim is None:
                self.head_dim = key_states.shape[-1]

            # Undo RoPE before prediction/quantization.
            if has_rotary_kwargs:
                key_states = apply_rotary_to_keys(key_states, cos=self.cos.to(device), sin=-self.sin.to(device))

            key_states, value_states = map(combine_heads, (key_states, value_states))

            if layer_idx == 0:
                if self.first_layer_quantizer:
                    self.quantized_first_layer_k_cache = self.first_layer_quantizer.quantize(
                        key_states.flatten(0, -2)
                    )
                    reconstructed_key_states = (
                        self.first_layer_quantizer.dequantize(self.quantized_first_layer_k_cache)
                        .view_as(key_states)
                        .to(dtype=dtype, device=device)
                    )
                    self.key_states_cache[0] = reconstructed_key_states

                    self.quantized_first_layer_v_cache = self.first_layer_quantizer.quantize(
                        value_states.flatten(0, -2)
                    )
                    reconstructed_value_states = (
                        self.first_layer_quantizer.dequantize(self.quantized_first_layer_v_cache)
                        .view_as(value_states)
                        .to(dtype=dtype, device=device)
                    )
                    self.value_states_cache[0] = reconstructed_value_states
                else:
                    reconstructed_key_states = self.key_states_cache[0] = key_states
                    reconstructed_value_states = self.value_states_cache[0] = value_states
            else:
                predicted_key_states = self.predict_next_key_states().to(device)
                self.key_states_cache[layer_idx] = self.quantizer.quantize(
                    (key_states - predicted_key_states).flatten(0, -2)
                )
                reconstructed_key_states = predicted_key_states + self.quantizer.dequantize(
                    self.key_states_cache[layer_idx]
                ).view_as(key_states).to(dtype=dtype, device=device)

                predicted_value_states = self.predict_next_value_states(reconstructed_key_states).to(device)
                self.value_states_cache[layer_idx] = self.quantizer.quantize(
                    (value_states - predicted_value_states).flatten(0, -2)
                )
                reconstructed_value_states = predicted_value_states + self.quantizer.dequantize(
                    self.value_states_cache[layer_idx]
                ).view_as(value_states).to(dtype=dtype, device=device)

            # Return original data since it's available — avoid quantization errors for write step.
            result_key, result_value = key_states_original, value_states_original

        else:  # read mode
            if layer_idx == 0:
                reconstructed_key_states = self.key_states_cache[0]
                reconstructed_value_states = self.value_states_cache[0]
                device, dtype = reconstructed_key_states.device, reconstructed_key_states.dtype
            else:
                dtype = key_states.dtype
                device = key_states.device
                reconstructed_key_states = self.predict_next_key_states().to(device)
                reconstructed_key_states += (
                    self.quantizer.dequantize(self.key_states_cache[layer_idx])
                    .view_as(reconstructed_key_states)
                    .to(dtype=dtype, device=device)
                )

                reconstructed_value_states = self.predict_next_value_states(reconstructed_key_states).to(device)
                reconstructed_value_states += (
                    self.quantizer.dequantize(self.value_states_cache[layer_idx])
                    .view_as(reconstructed_value_states)
                    .to(dtype=dtype, device=device)
                )

            assert self.head_dim is not None
            result_key_without_rotary = split_heads(reconstructed_key_states, self.head_dim)
            if self.sin is not None and self.cos is not None:
                result_key = apply_rotary_to_keys(
                    result_key_without_rotary, cos=self.cos.to(device), sin=self.sin.to(device)
                )
            else:
                result_key = result_key_without_rotary
            result_value = split_heads(reconstructed_value_states, self.head_dim)

        self.next_layer_idx = layer_idx + 1
        self.previous_key_reconstruction = reconstructed_key_states
        self.previous_value_reconstruction = reconstructed_value_states
        return result_key, result_value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_seq_length()})"
