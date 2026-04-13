"""
Model surgery utilities for calibration: layer enumeration and input capture.

Ported from aquakv/modelutils.py with cleaned imports.
"""
from itertools import chain
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import transformers
from tqdm import trange
from transformers import PreTrainedModel

LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere", "qwen2", "qwen3")
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")


def get_layers(model: PreTrainedModel) -> List[nn.Module]:
    """Return the list of transformer block modules for supported architectures."""
    model_type = model.config.model_type
    if model_type in (*LLAMA_LIKE, "phi3"):
        return model.model.layers
    elif model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(
            f"Unsupported model type {model_type} — supported: {LLAMA_LIKE + ('phi3',) + FALCON_TYPES + ('opt',)}"
        )


@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    """
    Mock model forward to capture inputs to the first transformer layer.

    :returns: (inps, forward_args) where inps is a list of activation tensors
        and forward_args contains attention_mask, position_ids, etc.
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data, torch.Tensor) and data.shape[0] == 1:
        assert data.ndim == 2
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")

    assert all(sequence.shape[1] == model_seqlen for sequence in data)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for name in forward_arg_names:
                cache[name] = kwargs.get(name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module
    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps)
    return inps, forward_args


def update_outs_inplace_(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    **kwargs,
):
    """Run a single layer over a large set of (possibly offloaded) inputs, update outs in-place."""
    assert len(inps) == len(outs) == len(devices)
    if len(devices) == 1:
        return _update_outs(layer, inps[0], outs[0], **kwargs)
    else:
        return _update_outs_parallel(devices, layer, inps, outs, **kwargs)


@torch.no_grad()
def _update_outs(
    layer: nn.Module,
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
            out_losses.append(outs_batch_loss.mean().item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def _update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [_update_outs for _ in devices]
    inputs_by_device = [(layer_replicas[i], inps[i], outs[i], compute_mse) for i in range(len(devices))]
    kwargs_by_device = [
        {k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in forward_args.items()}
        for i in range(len(devices))
    ]
    out_losses_by_device = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))
