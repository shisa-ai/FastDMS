"""
Cache orchestration: buffered prediction + quantization pipeline, prefix/suffix splitting.

PredictorHiggsCache: accumulates tokens in a BF16 buffer, then flushes to a quantized chunk
    with cross-layer prediction when the buffer fills.
TreatPrefixSeparately: routes first N tokens to a prefix cache and the rest to a suffix cache.
FrozenCache: materializes a quantized chunk to plain BF16 DynamicCache (for fast eval).

Ported from aquakv/cache_utils.py with cleaned imports.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers

from .quantizers import QuantizerBase
from .predictors import SingleChunkQuantizedCacheWithPredictors, split_heads, combine_heads


class TreatPrefixSeparately(transformers.cache_utils.Cache):
    """
    Routes the first prefix_size tokens to a prefix cache (typically unquantized DynamicCache)
    and all subsequent tokens to a suffix cache (typically PredictorHiggsCache).
    Attention always sees both parts concatenated.
    """

    def __init__(
        self,
        prefix_size: int,
        prefix_cache: transformers.cache_utils.Cache,
        suffix_cache: transformers.cache_utils.Cache,
    ):
        super().__init__(layers=[])
        self.prefix_size = prefix_size
        self.prefix_cache = prefix_cache
        self.suffix_cache = suffix_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_new_tokens = key_states.shape[-2]
        num_added_to_prefix = min(
            self.prefix_size - self.prefix_cache.get_seq_length(layer_idx), num_new_tokens
        )

        if num_added_to_prefix == num_new_tokens and num_new_tokens > 0:
            assert self.suffix_cache.get_seq_length(layer_idx) == 0
            return self.prefix_cache.update(key_states, value_states, layer_idx, cache_kwargs)

        else:
            if cache_kwargs and set(cache_kwargs.keys()) != {"sin", "cos", "cache_position"}:
                warnings.warn(f"{self.__class__.__name__} was not tested with {cache_kwargs=}")
            prefix_cache_kwargs = (
                {
                    k: (v[..., :num_added_to_prefix, :] if v.ndim > 1 else v[:num_added_to_prefix])
                    for k, v in cache_kwargs.items()
                }
                if cache_kwargs
                else cache_kwargs
            )
            suffix_cache_kwargs = (
                {
                    k: (v[..., num_added_to_prefix:, :] if v.ndim > 1 else v[num_added_to_prefix:])
                    for k, v in cache_kwargs.items()
                }
                if cache_kwargs
                else cache_kwargs
            )
            prefix_keys, prefix_values = self.prefix_cache.update(
                key_states[..., :num_added_to_prefix, :],
                value_states[..., :num_added_to_prefix, :],
                layer_idx,
                prefix_cache_kwargs,
            )
            suffix_keys, suffix_values = self.suffix_cache.update(
                key_states[..., num_added_to_prefix:, :],
                value_states[..., num_added_to_prefix:, :],
                layer_idx,
                suffix_cache_kwargs,
            )
            return (
                torch.cat([prefix_keys, suffix_keys], dim=-2),
                torch.cat([prefix_values, suffix_values], dim=-2),
            )

    def get_seq_length(self, **kwargs) -> int:
        return self.prefix_cache.get_seq_length(**kwargs) + self.suffix_cache.get_seq_length(**kwargs)


class PredictorHiggsCache(transformers.cache_utils.Cache):
    """
    Buffered cache with cross-layer predictors and residual HIGGS quantization.

    Accumulates tokens in a BF16 DynamicCache buffer. When the buffer reaches
    min_buffer_size tokens, flushes it to a SingleChunkQuantizedCacheWithPredictors.
    On read, concatenates all quantized chunks + the live buffer.

    :param config: model config (used for num_hidden_layers)
    :param make_quantized_cache: callable that returns a fresh SingleChunkQuantizedCacheWithPredictors
    :param min_buffer_size: flush buffer when it reaches this many tokens
    :param save_dequantized_values: if True, materialize chunks to BF16 FrozenCache after flush
        (faster eval but no memory savings — for backwards compatibility / prototyping)
    """

    def __init__(
        self,
        *,
        config: transformers.PretrainedConfig,
        make_quantized_cache: callable,
        min_buffer_size: int,
        save_dequantized_values: bool = False,
    ):
        super().__init__(layers=[])
        self.make_quantized_cache = make_quantized_cache
        self.save_dequantized_values = save_dequantized_values
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.buffer_size = min_buffer_size
        self.buffer_cache = transformers.cache_utils.DynamicCache()
        self.buffer_kwargs: list = []
        self.combined_buffer_kwargs = None

        self.quantized_caches: List[SingleChunkQuantizedCacheWithPredictors] = []
        self.latest_quantized_cache = self.make_quantized_cache()

        self.next_layer_idx = 0
        self.compressing = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx in (self.next_layer_idx, 0), (layer_idx, self.next_layer_idx, 0)
        if layer_idx == 0:
            self.buffer_kwargs.append(cache_kwargs)
            if cache_kwargs is not None:
                assert set(cache_kwargs.keys()) == {"sin", "cos", "cache_position"}

        empty = key_states[..., :0, :]
        key_buffer, value_buffer = self.buffer_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        empty_kwargs = (
            {k: (v[..., :0, :] if v.ndim > 1 else v[:0]) for k, v in cache_kwargs.items()}
            if cache_kwargs
            else cache_kwargs
        )
        dequantized_key_chunks, dequantized_value_chunks = zip(
            *[cache.update(empty, empty, layer_idx, empty_kwargs) for cache in self.quantized_caches]
            + [(key_buffer, value_buffer)]
        )

        actual_device = key_buffer.device
        dequantized_key_chunks = [k.to(actual_device) for k in dequantized_key_chunks]
        dequantized_value_chunks = [v.to(actual_device) for v in dequantized_value_chunks]

        combined_key_states = torch.cat(dequantized_key_chunks, dim=-2)
        combined_value_states = torch.cat(dequantized_value_chunks, dim=-2)

        if not self.compressing and key_buffer.shape[-2] >= self.buffer_size:
            self.compressing = True
            self.combined_buffer_kwargs = self._combine_buffer_kwargs()
            self.buffer_kwargs = []

        if self.compressing:
            self.latest_quantized_cache.update(
                key_buffer, value_buffer, layer_idx, self.combined_buffer_kwargs
            )
            if hasattr(self.buffer_cache, "key_cache"):
                self.buffer_cache.key_cache[layer_idx] = empty.clone()
                self.buffer_cache.value_cache[layer_idx] = empty.clone()
            else:
                layer = self.buffer_cache.layers[layer_idx]
                layer.keys = empty.clone()
                layer.values = empty.clone()

        if self.compressing and layer_idx == self.num_layers - 1:
            cache_to_save = self.latest_quantized_cache
            if self.save_dequantized_values:
                cache_to_save = FrozenCache(cache_to_save, config=self.config)
            self.quantized_caches.append(cache_to_save)

            self.latest_quantized_cache = self.make_quantized_cache()
            self.buffer_cache = transformers.cache_utils.DynamicCache()
            self.combined_buffer_kwargs = None
            self.compressing = False

        self.next_layer_idx = layer_idx + 1
        return combined_key_states, combined_value_states

    def _combine_buffer_kwargs(self):
        assert len(self.buffer_kwargs) > 0
        if any(kw is None for kw in self.buffer_kwargs):
            return None
        return dict(
            cos=torch.cat([kw["cos"] for kw in self.buffer_kwargs], dim=-2),
            sin=torch.cat([kw["sin"] for kw in self.buffer_kwargs], dim=-2),
            cache_position=torch.cat([kw["cache_position"] for kw in self.buffer_kwargs], dim=0),
        )

    def get_seq_length(self, *args, **kwargs) -> int:
        return self.buffer_cache.get_seq_length(*args, **kwargs) + sum(
            qc.get_seq_length(*args, **kwargs) for qc in self.quantized_caches
        )


class FrozenCache(transformers.cache_utils.DynamicCache):
    """
    Materializes a SingleChunkQuantizedCacheWithPredictors into a plain BF16 DynamicCache
    by dequantizing all layers once. Used for faster eval when dequantization is slow.
    """

    def __init__(self, cache: SingleChunkQuantizedCacheWithPredictors, config: transformers.PretrainedConfig):
        super().__init__()
        batch_size = cache.key_states_cache[0].shape[0]
        device, dtype = cache.key_states_cache[0].device, cache.key_states_cache[0].dtype
        cache_length = cache.get_seq_length()
        cache_position = torch.arange(cache_length, device=device)
        past_key_values = _get_past_key_values(
            cache=cache, config=config, batch_size=batch_size, device=device, dtype=dtype
        )
        for layer_idx in range(len(past_key_values)):
            super().update(
                *past_key_values[layer_idx],
                layer_idx=layer_idx,
                cache_kwargs=dict(cache_position=cache_position),
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert key_states.numel() == value_states.numel() == 0
        if hasattr(self, "key_cache"):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        layer = self.layers[layer_idx]
        return layer.keys, layer.values


@torch.no_grad()
def _get_past_key_values(
    cache: transformers.cache_utils.Cache,
    config: transformers.PretrainedConfig,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim
    elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        head_dim = config.hidden_size // config.num_attention_heads
    else:
        raise RuntimeError("Head dim cannot be inferred from config.")

    empty = torch.zeros(batch_size, config.num_key_value_heads, 0, head_dim, device=device, dtype=dtype)
    empty_rotary = torch.zeros(batch_size, config.num_key_value_heads, 0, head_dim, device=device, dtype=dtype)

    past_key_values = []
    for layer_idx in range(config.num_hidden_layers):
        past_key_values.append(
            cache.update(
                empty,
                empty,
                layer_idx=layer_idx,
                cache_kwargs=dict(cos=empty_rotary, sin=empty_rotary, cache_position=torch.arange(0, device=device)),
            )
        )
    return past_key_values
