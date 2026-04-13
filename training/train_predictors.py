"""
Train AQUA-KV cross-layer predictors for a given model.

Each predictor is a single nn.Linear that predicts the current layer's keys (or values)
from the previous layer's reconstructed keys (or [keys, values]). Predictors are trained
via closed-form OLS on calibration data.

Usage:
    python -m shisa_kvquant.train_predictors \
        --model_name Qwen/Qwen3-8B \
        --dataset wikitext2 \
        --edenn_d 2 --edenn_n 256 \
        --predictors_output_path predictors.pt

Ported from aquakv/train_predictors.py with shisa_kvquant imports.
"""
import math
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import transformers
from tqdm import trange

from . import data_utils, model_utils
from .quantizers import QuantizerBase, HiggsQuantizer
from .linear_utils import fit_linear_regression


DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    None: None,
}


class OutputCatcher(nn.Module):
    """Wraps a module to capture its output tensors from forward passes."""

    def __init__(self, inner: nn.Module, offload_activations: bool):
        super().__init__()
        self.inner = inner
        self.offload_activations = offload_activations
        self.outputs: list = []

    def forward(self, inp, **kwargs):
        output = self.inner(inp)
        self.outputs.append(output.to("cpu" if self.offload_activations else inp.device, copy=True))
        return output


def flatten_key_sample(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.flatten(-2)
    return tensor


def get_predictor(
    args: argparse.Namespace,
    predictor_inputs: List[torch.Tensor],
    targets: List[torch.Tensor],
) -> Tuple[nn.Module, float, float]:
    """Train a linear predictor, return (predictor, train_mse, valid_mse)."""
    device = args.devices[0]
    X = torch.stack(predictor_inputs, dim=0)
    Y = torch.stack(targets, dim=0)
    assert X.shape[:-1] == Y.shape[:-1] == (args.total_nsamples, args.model_seqlen)

    train_ids, valid_ids = torch.randperm(
        len(X), generator=torch.Generator(X.device).manual_seed(args.seed), device=X.device
    ).split_with_sizes((args.total_nsamples - args.valid_nsamples, args.valid_nsamples))

    X_train, X_valid = X[train_ids].flatten(0, -2), X[valid_ids].flatten(0, -2)
    Y_train, Y_valid = Y[train_ids].flatten(0, -2), Y[valid_ids].flatten(0, -2)

    weight, bias = fit_linear_regression(
        X_train, Y_train,
        reg_rate=args.percdamp, fit_intercept=True,
        compute_device=device, compute_dtype=torch.float32, chunk_size=args.chunk_size,
    )
    predictor = nn.Linear(*weight.shape[::-1], dtype=X.dtype, device=device)
    with torch.no_grad():
        predictor.weight[...] = weight
        predictor.bias[...] = bias

    mse_train = compute_relative_mse(predictor, X_train, Y_train, compute_device=device, chunk_size=args.chunk_size)
    mse_valid = compute_relative_mse(predictor, X_valid, Y_valid, compute_device=device, chunk_size=args.chunk_size)
    return predictor, mse_train, mse_valid


@torch.no_grad()
def get_dequant_values(
    args: argparse.Namespace,
    quantizer: QuantizerBase,
    predictor: Optional[nn.Module],
    predictor_inputs: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
):
    """Return reconstructed (predict + quantize-dequantize residual) tensors."""
    assert len(predictor_inputs) == len(values)
    values_dequantized = []
    for i in trange(len(predictor_inputs), desc="get_dequant_values", leave=False):
        inputs_i = predictor_inputs[i].to(args.devices[0], non_blocking=True)
        values_i = values[i].to(args.devices[0], non_blocking=True)
        if predictor is None:
            pred_i = torch.zeros_like(values_i)
        else:
            pred_i = predictor(inputs_i)
        delta = values_i - pred_i
        delta_dq = quantizer.quantize_dequantize(delta.flatten(0, -2)).reshape(delta.shape)
        values_dequantized.append((delta_dq + pred_i).to(values[i].device, non_blocking=True))
    return values_dequantized


@torch.no_grad()
def compute_relative_mse(
    predictor: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    chunk_size: Optional[int] = None,
    compute_device: Optional[torch.device] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> float:
    """Compute ||predictor(X) - Y||^2 / ||Y||^2."""
    if compute_device is None:
        compute_device = next(predictor.parameters()).device
    if chunk_size is None:
        return ((predictor(X) - Y).norm() / Y.norm()).item() ** 2
    numerator = denominator = 0
    for chunk_start in trange(0, len(X), chunk_size, desc="compute_relative_mse", leave=False):
        xb, yb = [
            tensor[chunk_start : chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True
            )
            for tensor in (X, Y)
        ]
        numerator += (predictor(xb) - yb).norm().square().item()
        denominator += yb.norm().square().item()
    return numerator / denominator


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Train AQUA-KV cross-layer predictors")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset: wikitext2, c4, ptb, pajama, or file path")
    parser.add_argument("--edenn_d", type=int, required=True, help="EDEN grid dimension (2)")
    parser.add_argument("--edenn_n", type=int, required=True, help="EDEN grid size: 16=2bit, 64=3bit, 256=4bit")
    parser.add_argument("--not_quantize_first_layer", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--compute_dtype", type=str, default=None)
    parser.add_argument("--model_seqlen", type=int, default=8192)
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--total_nsamples", type=int, default=256)
    parser.add_argument("--valid_nsamples", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--percdamp", type=float, default=1e-3)
    parser.add_argument("--hadamard_groupsize", type=int, default=1024)
    parser.add_argument("--predictors_output_path", type=str, default="./key_value_predictors.pt")
    return parser


def main():
    parser = make_arg_parser()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    args = parser.parse_args()

    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(d) for d in args.devices]
    assert len(args.devices) == 1, "training-time parallelism is not implemented yet"

    args.torch_dtype = DTYPE_MAP[args.torch_dtype]
    args.compute_dtype = DTYPE_MAP.get(args.compute_dtype, args.compute_dtype)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True,
        use_cache=False, trust_remote_code=True,
    )
    config = transformers.AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    data = data_utils.get_loaders(
        args.dataset, nsamples=args.total_nsamples, seed=args.seed,
        model_path=args.model_name, seqlen=args.model_seqlen,
    )

    common_kwargs = dict(
        hadamard_groupsize=args.hadamard_groupsize,
        channel_size=(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            * config.num_key_value_heads
        ),
    )
    quantizer = HiggsQuantizer(edenn_d=args.edenn_d, edenn_n=args.edenn_n, **common_kwargs)
    first_layer_quantizer = (
        None if args.not_quantize_first_layer
        else HiggsQuantizer(edenn_d=2, edenn_n=256, **common_kwargs)
    )

    layers = model_utils.get_layers(model)
    inps, forward_args = model_utils.get_inps(model, data, args.model_seqlen, args.devices, args.offload_activations)

    for k, v in forward_args.items():
        forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

    model.model.rotary_emb.to(args.devices[0])
    forward_args["position_embeddings"] = model.model.rotary_emb.forward(
        inps[0][:1].to(args.devices[0]), torch.arange(0, args.model_seqlen).unsqueeze(0).to(args.devices[0])
    )
    model.model.rotary_emb.cpu()

    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    old_attn_keys = None
    old_attn_values = None

    key_predictors = {}
    value_predictors = {}

    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        layer_device_original = next(layers[layer_index].parameters()).device
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        layer = layers[layer_index].to(device=args.devices[0], dtype=args.compute_dtype or layer_dtype_original)

        key_capture_name = "k_norm" if hasattr(layer.self_attn, "k_norm") else "k_proj"
        key_capture_module = getattr(layer.self_attn, key_capture_name)
        setattr(layer.self_attn, key_capture_name, OutputCatcher(key_capture_module, args.offload_activations))
        layer.self_attn.v_proj = OutputCatcher(layer.self_attn.v_proj, args.offload_activations)

        model_utils.update_outs_inplace_(args.devices, layer, inps, outs, **forward_args, compute_mse=False)

        key_capture = getattr(layer.self_attn, key_capture_name)
        attn_keys = [flatten_key_sample(elem[0]) for elem in key_capture.outputs]
        attn_values = [elem[0] for elem in layer.self_attn.v_proj.outputs]

        setattr(layer.self_attn, key_capture_name, key_capture.inner)
        layer.self_attn.v_proj = layer.self_attn.v_proj.inner

        layers[layer_index] = layer.to(device=layer_device_original, dtype=layer_dtype_original)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        if layer_index == 0:
            old_attn_keys = attn_keys
            old_attn_values = attn_values
            if args.not_quantize_first_layer:
                print("Not quantizing first layer")
                continue

        key_predictor_inputs = list(old_attn_keys)

        if layer_index == 0:
            key_predictor, mse_train_keys, mse_valid_keys = None, 10000, 10000
        else:
            key_predictor, mse_train_keys, mse_valid_keys = get_predictor(args, key_predictor_inputs, attn_keys)

        attn_keys = get_dequant_values(
            args, quantizer if layer_index != 0 else first_layer_quantizer,
            key_predictor, key_predictor_inputs, attn_keys,
        )
        del key_predictor_inputs
        if layer_index != 0:
            key_predictors[layer_index] = key_predictor.cpu()

        train_bits_keys = -math.log(mse_train_keys) / math.log(4)
        valid_bits_keys = -math.log(mse_valid_keys) / math.log(4)
        print(
            f"{layer_index=}\tPREDICTOR_KEYS   \t| relMSE train: {mse_train_keys:.4f} valid: {mse_valid_keys:.4f} "
            f"| equiv.bits train: {train_bits_keys:.2f} valid: {valid_bits_keys:.2f}"
        )

        value_predictor_inputs = [
            torch.cat([k_i, old_v_i], dim=-1) for k_i, old_v_i in zip(attn_keys, old_attn_values)
        ]
        if layer_index == 0:
            value_predictor, mse_train_values, mse_valid_values = None, 10000, 10000
        else:
            value_predictor, mse_train_values, mse_valid_values = get_predictor(
                args, value_predictor_inputs, attn_values
            )

        attn_values = get_dequant_values(
            args, quantizer if layer_index != 0 else first_layer_quantizer,
            value_predictor, value_predictor_inputs, attn_values,
        )
        if layer_index != 0:
            value_predictors[layer_index] = value_predictor.cpu()
        del value_predictor_inputs

        train_bits_values = -math.log(mse_train_values) / math.log(4)
        valid_bits_values = -math.log(mse_valid_values) / math.log(4)
        print(
            f"{layer_index=}\tPREDICTOR_VALUES \t| relMSE train: {mse_train_values:.4f} valid: {mse_valid_values:.4f} "
            f"| equiv.bits train: {train_bits_values:.2f} valid: {valid_bits_values:.2f}"
        )

        old_attn_keys, old_attn_values = attn_keys, attn_values

    Path(args.predictors_output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(key_predictors=key_predictors, value_predictors=value_predictors), args.predictors_output_path)
    print("Saved predictors to", args.predictors_output_path)


if __name__ == "__main__":
    main()
