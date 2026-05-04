"""
DMS inference evaluation: measure PPL with hard eviction decisions active.

Loads a DMS-trained checkpoint and evaluates with actual token eviction.
The borrowed neuron in each layer's q_proj provides binary eviction decisions:
  decision = (q_proj(h)[-1] * alpha_scale - alpha_offset) > 0
Evicted tokens outside the sliding window are masked from attention.

Supports optional HIGGS quantization of surviving KV entries via --higgs-bits.

Usage:
    python -m training.dms_eval \
        --model results/dms/llama32-1b-cr8-v4-full/final \
        --vanilla-model meta-llama/Llama-3.2-1B \
        --device cuda:1

    # DMS + HIGGS 4-bit composition:
    python -m training.dms_eval \
        --model results/dms/llama32-1b-cr8-v4-full/final \
        --vanilla-model meta-llama/Llama-3.2-1B \
        --higgs-bits 4 \
        --device cuda:1
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try:
    from .dms_mask import dms_outside_window_mask
except ImportError:  # pragma: no cover - direct script execution fallback
    from dms_mask import dms_outside_window_mask


def dms_inference_forward(
    model,
    input_ids: torch.Tensor,
    alpha_scale: float,
    alpha_offset: float,
    window_size: int,
    higgs_quantizer=None,
    aqua_predictors=None,
) -> torch.Tensor:
    """
    Forward pass with hard DMS eviction decisions and optional HIGGS quantization.

    For each attention layer, extracts the borrowed neuron's decision,
    builds a hard attention mask that removes evicted tokens outside
    the sliding window, and injects it into the attention computation.

    If higgs_quantizer is provided, also quantizes/dequantizes the K and V
    states to simulate HIGGS packed storage quality impact.

    Returns (logits, eviction_stats_dict).
    """
    device = input_ids.device
    batch, seq_len = input_ids.shape
    config = model.config
    num_kv_heads = config.num_key_value_heads
    num_heads = config.num_attention_heads
    q_per_kv = num_heads // num_kv_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    hooks = []
    eviction_counts = []

    # Pre-compute window mask.
    outside_window = dms_outside_window_mask(seq_len, window_size, device=device).float()

    def make_attn_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return

            B, T, D = hidden_states.shape

            # Extract decision logits from borrowed neuron
            q_full = module.q_proj(hidden_states)
            q_reshaped = q_full.view(B, T, -1, head_dim).transpose(1, 2)

            if hasattr(module, "q_norm") and module.q_norm is not None:
                q_reshaped = module.q_norm(q_reshaped)

            raw_logits = q_reshaped[:, ::q_per_kv, :, -1]
            decision_logits = raw_logits * alpha_scale - alpha_offset

            # Hard binary decisions: > 0 means evict
            decisions = (decision_logits > 0).float()
            eviction_counts.append(decisions.sum().item())

            # Build hard attention mask
            eviction_per_key = decisions.unsqueeze(2)  # [B, H, 1, T]
            ow = outside_window[:T, :T].unsqueeze(0).unsqueeze(0)
            # mask = 0 where evicted AND outside window, 1 otherwise
            mask_attn = 1.0 - eviction_per_key * ow

            # Convert to additive mask
            additive_mask = torch.where(
                mask_attn > 0.5,
                torch.zeros_like(mask_attn),
                torch.full_like(mask_attn, float("-inf")),
            )

            # Combine with causal mask
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )
            combined_mask = additive_mask + causal_mask.unsqueeze(0).unsqueeze(0)

            # Expand to num_heads
            expanded_mask = combined_mask.repeat_interleave(q_per_kv, dim=1)
            kwargs["attention_mask"] = expanded_mask.to(hidden_states.dtype)

            return args, kwargs

        return hook_fn

    # AQUA + HIGGS quantization state: track previous layer's reconstructed K,V for prediction
    aqua_state = {"prev_k": None, "prev_v": None}

    def make_kv_quant_hook(layer_idx, is_key=True):
        """Post-hook on k_proj/v_proj that applies AQUA prediction + HIGGS residual quantization."""
        def hook_fn(module, input, output):
            dtype = output.dtype
            B, T, C = output.shape
            flat = output.reshape(-1, C)

            if aqua_predictors is not None and layer_idx > 0:
                # AQUA: predict from previous layer, quantize residual
                if is_key:
                    pred = aqua_predictors["key_predictors"][layer_idx]
                    pred = pred.to(flat.device)
                    predicted = pred(aqua_state["prev_k"])
                else:
                    pred = aqua_predictors["value_predictors"][layer_idx]
                    pred = pred.to(flat.device)
                    # Value predictor input = [reconstructed_keys, prev_values]
                    pred_input = torch.cat([aqua_state["cur_k_recon"], aqua_state["prev_v"]], dim=-1)
                    predicted = pred(pred_input)

                residual = flat - predicted
                if higgs_quantizer is not None:
                    residual_deq = higgs_quantizer.quantize_dequantize(residual)
                else:
                    residual_deq = residual
                reconstructed = (predicted + residual_deq).to(dtype=dtype)
            elif higgs_quantizer is not None:
                # HIGGS only (no AQUA)
                reconstructed = higgs_quantizer.quantize_dequantize(flat).to(dtype=dtype)
            else:
                reconstructed = flat.to(dtype=dtype)

            # Update AQUA state for next layer
            if is_key:
                aqua_state["cur_k_recon"] = reconstructed.detach()
            else:
                # After both K and V are processed, update prev for next layer
                aqua_state["prev_k"] = aqua_state["cur_k_recon"]
                aqua_state["prev_v"] = reconstructed.detach()

            return reconstructed.reshape(B, T, C)
        return hook_fn

    # Register hooks
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        hooks.append(attn.register_forward_pre_hook(make_attn_pre_hook(layer_idx), with_kwargs=True))
        if higgs_quantizer is not None or aqua_predictors is not None:
            hooks.append(attn.k_proj.register_forward_hook(make_kv_quant_hook(layer_idx, is_key=True)))
            hooks.append(attn.v_proj.register_forward_hook(make_kv_quant_hook(layer_idx, is_key=False)))

    # Forward
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    # Clean up
    for h in hooks:
        h.remove()

    total_evictions = sum(eviction_counts)
    total_decisions = num_kv_heads * seq_len * batch * len(model.model.layers)
    eviction_rate = total_evictions / max(1, total_decisions)

    return outputs.logits, {"eviction_rate": eviction_rate, "total_evictions": total_evictions}


def evaluate_ppl(
    model,
    tokenizer,
    n_chunks: int = 8,
    seq_len: int = 512,
    device: str = "cuda:1",
    use_dms: bool = False,
    alpha_scale: float = 100.0,
    alpha_offset: float = 5.0,
    window_size: int = 256,
    higgs_quantizer=None,
    aqua_predictors=None,
):
    """Evaluate PPL on WikiText-2 validation."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]

    total_nll = 0.0
    total_tokens = 0
    total_eviction_rate = 0.0

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)

            if use_dms or higgs_quantizer is not None or aqua_predictors is not None:
                logits, stats = dms_inference_forward(
                    model, chunk,
                    alpha_scale=alpha_scale if use_dms else 0.0,
                    alpha_offset=alpha_offset if use_dms else 999.0,
                    window_size=window_size,
                    higgs_quantizer=higgs_quantizer,
                    aqua_predictors=aqua_predictors,
                )
                total_eviction_rate += stats["eviction_rate"]
            else:
                logits = model(chunk).logits

            shift_logits = logits[:, :-1].float()
            shift_labels = chunk[:, 1:]
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += shift_labels.numel()

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    avg_eviction_rate = total_eviction_rate / n_chunks if use_dms else 0.0
    return ppl, avg_eviction_rate


def make_higgs_quantizer(bits: int, channel_size: int):
    """Create a HiggsQuantizer for the given bit-width and channel size."""
    try:
        from .quantizers import HiggsQuantizer
    except ImportError:  # pragma: no cover - direct script execution fallback
        from quantizers import HiggsQuantizer

    edenn_n = {2: 16, 3: 64, 4: 256}[bits]
    return HiggsQuantizer(
        hadamard_groupsize=channel_size,
        edenn_d=2,
        edenn_n=edenn_n,
        channel_size=channel_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate DMS-trained model with hard eviction")
    parser.add_argument("--model", type=str, required=True, help="Path to DMS-trained checkpoint")
    parser.add_argument("--vanilla-model", type=str, default="meta-llama/Llama-3.2-1B", help="Vanilla model for comparison")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--alpha-scale", type=float, default=100.0)
    parser.add_argument("--alpha-offset", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--higgs-bits", type=int, default=0, choices=[0, 2, 3, 4], help="HIGGS quantization bits (0=none)")
    parser.add_argument("--aqua-predictors", type=str, default=None, help="Path to AQUA predictor .pt file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.vanilla_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up HIGGS quantizer if requested
    higgs_q = None
    if args.higgs_bits > 0:
        config = AutoModelForCausalLM.from_pretrained(args.vanilla_model, dtype=torch.bfloat16, device_map={"": args.device}).config
        kv_channel_size = config.num_key_value_heads * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        higgs_q = make_higgs_quantizer(args.higgs_bits, kv_channel_size)
        print(f"HIGGS {args.higgs_bits}-bit quantizer: channel_size={kv_channel_size}")
        torch.cuda.empty_cache()

    # Load AQUA predictors if requested
    aqua_preds = None
    if args.aqua_predictors:
        aqua_preds = torch.load(args.aqua_predictors, weights_only=False, map_location=args.device)
        print(f"AQUA predictors loaded: {len(aqua_preds.get('key_predictors', {}))} key, {len(aqua_preds.get('value_predictors', {}))} value")

    # Evaluate vanilla model
    print(f"Loading vanilla model: {args.vanilla_model}")
    vanilla = AutoModelForCausalLM.from_pretrained(
        args.vanilla_model, dtype=torch.bfloat16, device_map={"": args.device}
    )
    vanilla.eval()
    print("Evaluating vanilla PPL...")
    ppl_vanilla, _ = evaluate_ppl(vanilla, tokenizer, args.n_chunks, args.seq_len, args.device)
    print(f"  Vanilla PPL: {ppl_vanilla:.4f}")
    del vanilla
    torch.cuda.empty_cache()

    # Evaluate DMS model without eviction (quality floor)
    print(f"\nLoading DMS model: {args.model}")
    dms = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device}
    )
    dms.eval()
    print("Evaluating DMS without eviction (quality floor)...")
    ppl_dms_no_evict, _ = evaluate_ppl(dms, tokenizer, args.n_chunks, args.seq_len, args.device)
    delta_no_evict = (ppl_dms_no_evict - ppl_vanilla) / ppl_vanilla * 100
    print(f"  DMS (no eviction) PPL: {ppl_dms_no_evict:.4f} ({delta_no_evict:+.2f}%)")

    # Evaluate DMS model with hard eviction
    print("Evaluating DMS WITH hard eviction...")
    ppl_dms_evict, evict_rate = evaluate_ppl(
        dms, tokenizer, args.n_chunks, args.seq_len, args.device,
        use_dms=True,
        alpha_scale=args.alpha_scale,
        alpha_offset=args.alpha_offset,
        window_size=args.window_size,
    )
    delta_evict = (ppl_dms_evict - ppl_vanilla) / ppl_vanilla * 100
    print(f"  DMS (with eviction) PPL: {ppl_dms_evict:.4f} ({delta_evict:+.2f}%)")
    print(f"  Eviction rate: {evict_rate:.4f}")

    # Evaluate DMS + HIGGS if requested
    ppl_dms_higgs = None
    delta_higgs = None
    if higgs_q is not None:
        print(f"\nEvaluating DMS + HIGGS {args.higgs_bits}-bit...")
        ppl_dms_higgs, evict_rate_higgs = evaluate_ppl(
            dms, tokenizer, args.n_chunks, args.seq_len, args.device,
            use_dms=True,
            alpha_scale=args.alpha_scale,
            alpha_offset=args.alpha_offset,
            window_size=args.window_size,
            higgs_quantizer=higgs_q,
        )
        delta_higgs = (ppl_dms_higgs - ppl_vanilla) / ppl_vanilla * 100
        bit_cr = 16 / args.higgs_bits  # BF16 → N-bit compression
        total_cr = 1 / (1 - evict_rate_higgs) * bit_cr if evict_rate_higgs < 1 else float("inf")
        print(f"  DMS + HIGGS {args.higgs_bits}b PPL: {ppl_dms_higgs:.4f} ({delta_higgs:+.2f}%)")
        print(f"  Total effective CR: {total_cr:.1f}x (DMS {1/(1-evict_rate_higgs):.1f}x × HIGGS {bit_cr:.1f}x)")

    # Evaluate DMS + AQUA (no HIGGS) if AQUA predictors provided
    ppl_dms_aqua = None
    delta_dms_aqua = None
    if aqua_preds is not None:
        print(f"\nEvaluating DMS + AQUA (no bit quantization)...")
        ppl_dms_aqua, _ = evaluate_ppl(
            dms, tokenizer, args.n_chunks, args.seq_len, args.device,
            use_dms=True,
            alpha_scale=args.alpha_scale,
            alpha_offset=args.alpha_offset,
            window_size=args.window_size,
            aqua_predictors=aqua_preds,
        )
        delta_dms_aqua = (ppl_dms_aqua - ppl_vanilla) / ppl_vanilla * 100
        print(f"  DMS + AQUA PPL: {ppl_dms_aqua:.4f} ({delta_dms_aqua:+.2f}%)")

    # Evaluate DMS + AQUA + HIGGS (triple composition)
    ppl_triple = None
    delta_triple = None
    if aqua_preds is not None and higgs_q is not None:
        print(f"\nEvaluating DMS + AQUA + HIGGS {args.higgs_bits}-bit (triple composition)...")
        ppl_triple, evict_rate_triple = evaluate_ppl(
            dms, tokenizer, args.n_chunks, args.seq_len, args.device,
            use_dms=True,
            alpha_scale=args.alpha_scale,
            alpha_offset=args.alpha_offset,
            window_size=args.window_size,
            higgs_quantizer=higgs_q,
            aqua_predictors=aqua_preds,
        )
        delta_triple = (ppl_triple - ppl_vanilla) / ppl_vanilla * 100
        bit_cr = 16 / args.higgs_bits
        total_cr_triple = 1 / (1 - evict_rate_triple) * bit_cr if evict_rate_triple < 1 else float("inf")
        print(f"  DMS + AQUA + HIGGS {args.higgs_bits}b PPL: {ppl_triple:.4f} ({delta_triple:+.2f}%)")
        print(f"  Total effective CR: {total_cr_triple:.1f}x")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Vanilla PPL:          {ppl_vanilla:.4f}")
    print(f"DMS (no eviction):    {ppl_dms_no_evict:.4f} ({delta_no_evict:+.2f}%)")
    print(f"DMS (with eviction):  {ppl_dms_evict:.4f} ({delta_evict:+.2f}%)")
    print(f"Eviction rate:        {evict_rate:.4f}")
    print(f"Effective CR:         {1/(1 - evict_rate):.1f}x" if evict_rate < 1 else "")
    if ppl_dms_higgs is not None:
        print(f"DMS + HIGGS {args.higgs_bits}b:     {ppl_dms_higgs:.4f} ({delta_higgs:+.2f}%)")
        print(f"Total CR:             {total_cr:.1f}x")
    if ppl_dms_aqua is not None:
        print(f"DMS + AQUA:           {ppl_dms_aqua:.4f} ({delta_dms_aqua:+.2f}%)")
    if ppl_triple is not None:
        print(f"DMS+AQUA+HIGGS {args.higgs_bits}b:  {ppl_triple:.4f} ({delta_triple:+.2f}%)")
        print(f"Triple CR:            {total_cr_triple:.1f}x")

    results = {
        "model": args.model,
        "vanilla_model": args.vanilla_model,
        "ppl_vanilla": ppl_vanilla,
        "ppl_dms_no_eviction": ppl_dms_no_evict,
        "ppl_dms_with_eviction": ppl_dms_evict,
        "delta_no_eviction_pct": delta_no_evict,
        "delta_with_eviction_pct": delta_evict,
        "eviction_rate": evict_rate,
        "effective_cr": 1 / (1 - evict_rate) if evict_rate < 1 else float("inf"),
        "alpha_scale": args.alpha_scale,
        "alpha_offset": args.alpha_offset,
        "window_size": args.window_size,
        "higgs_bits": args.higgs_bits,
        "n_chunks": args.n_chunks,
        "seq_len": args.seq_len,
    }
    if ppl_dms_higgs is not None:
        results["ppl_dms_higgs"] = ppl_dms_higgs
        results["delta_dms_higgs_pct"] = delta_higgs
        results["total_cr"] = total_cr
    if ppl_dms_aqua is not None:
        results["ppl_dms_aqua"] = ppl_dms_aqua
        results["delta_dms_aqua_pct"] = delta_dms_aqua
    if ppl_triple is not None:
        results["ppl_dms_aqua_higgs"] = ppl_triple
        results["delta_dms_aqua_higgs_pct"] = delta_triple
        results["total_cr_triple"] = total_cr_triple

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
