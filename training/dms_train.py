"""
DMS (Dynamic Memory Sparsification) training for KV cache token eviction.

Implements the training procedure from "Inference-Time Hyper-Scaling with KV Cache Compression"
(NeurIPS 2025, arXiv:2506.05345). No public training code exists — this is implemented from
the paper description and the inference code in nvidia/Qwen3-8B-DMS-8x.

Two-phase training:
  Phase 1 (neuron zeroing): Gradually zero the borrowed query dimension over 2000 steps
    so the model adapts to losing that dimension without quality collapse.
  Phase 2 (DMS retrofitting): Train eviction decisions with Gumbel-sigmoid and logit
    distillation. CR annealed linearly from 1 to target_cr over 100 steps per CR unit.

Usage:
    python -m shisa_kvquant.dms_train \
        --model Qwen/Qwen3.5-0.8B \
        --target-cr 8 \
        --context-len 4096 \
        --device cuda:1 \
        --output-dir results/dms/qwen35-08b-cr8
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

try:
    from .dms_mask import dms_outside_window_mask
except ImportError:  # pragma: no cover - direct script execution fallback
    from dms_mask import dms_outside_window_mask


# ---------------------------------------------------------------------------
# Gumbel-Sigmoid (differentiable binary decisions)
# ---------------------------------------------------------------------------

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Sigmoid: differentiable approximation to Bernoulli sampling.

    Returns soft probabilities in [0, 1]. With hard=True, returns hard 0/1
    with straight-through gradients.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-20, 1.0)))
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft  # straight-through
    return y_soft


# ---------------------------------------------------------------------------
# DMS Forward Pass (training version — with soft attention masking)
# ---------------------------------------------------------------------------

def dms_forward_with_masking(
    model: nn.Module,
    input_ids: torch.Tensor,
    alpha_scale: float,
    alpha_offset: float,
    tau: float,
    window_size: int,
    q_per_kv: int,
    hard: bool = False,
    cap_live_rates: torch.Tensor | None = None,
    cap_loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, dict[str, float]]:
    """
    Forward pass with DMS soft attention masking for training.

    Hooks into each attention layer to:
    1. Extract decision logits from the borrowed query dimension
    2. Zero the borrowed dimension (so it doesn't affect attention scores)
    3. Build a smooth attention mask from eviction decisions
    4. Inject the mask so attention uses it

    Returns (logits, decision_sum, total_decision_elements, cap_loss, cap_stats).
    Both logits and decision_sum are differentiable through the model's q_proj weights,
    so KD loss and compression loss both provide gradients to the decision head.
    """
    device = input_ids.device
    batch, seq_len = input_ids.shape
    config = model.config
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # Accumulators for decision stats
    decision_sum_parts = []
    cap_loss_parts = []
    cap_excess_parts = []
    total_elements = 0
    hooks = []

    # outside_window[q, k] = 1 if key token k is outside query q's DMS window.
    outside_window = dms_outside_window_mask(seq_len, window_size, device=device).float()

    def make_attn_pre_hook(layer_idx):
        """Pre-hook on the attention module to inject DMS masking."""
        def hook_fn(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return

            B, T, D = hidden_states.shape

            # Run q_proj to extract the borrowed neuron's decision logit
            q_full = module.q_proj(hidden_states)  # [B, T, num_heads * head_dim]
            q_reshaped = q_full.view(B, T, -1, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

            # Apply q_norm if present (Llama 3.2 has q_norm)
            if hasattr(module, "q_norm") and module.q_norm is not None:
                q_reshaped = module.q_norm(q_reshaped)

            # Extract last dim of first query head per GQA group
            raw_logits = q_reshaped[:, ::q_per_kv, :, -1]  # [B, num_kv_heads, T]
            decision_logits = raw_logits * alpha_scale - alpha_offset

            # Gumbel-sigmoid for differentiable decisions
            decisions = gumbel_sigmoid(decision_logits, tau=tau, hard=hard)  # [B, num_kv_heads, T]
            decision_sum_parts.append(decisions.sum())

            if cap_live_rates is not None and cap_loss_weights is not None:
                target_live = cap_live_rates[layer_idx, :num_kv_heads].to(
                    device=decisions.device, dtype=decisions.dtype
                )
                weights = cap_loss_weights[layer_idx, :num_kv_heads].to(
                    device=decisions.device, dtype=decisions.dtype
                )
                live_rate = (1.0 - decisions).mean(dim=(0, 2))  # [num_kv_heads]
                excess = F.relu(live_rate - target_live)
                active = (weights > 0).sum().clamp(min=1).to(dtype=decisions.dtype)
                cap_loss_parts.append((excess.square() * weights).sum() / active)
                cap_excess_parts.append((excess * weights).sum() / active)

            # Build soft attention mask from decisions.
            # eviction_prob_per_key[b, h, k] = probability that key token k will be evicted.
            # This is applied only when k is outside the sliding window of query q.
            eviction_per_key = decisions.unsqueeze(2)  # [B, H, 1, T]
            ow = outside_window[:T, :T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            # mask_attn[b, h, q, k] = 1 - evict_prob if outside window, else 1
            mask_attn = 1.0 - eviction_per_key * ow  # [B, H, T, T]

            # Convert to additive mask for SDPA: log(mask_attn) where 0→-inf, 1→0
            # Clamp to avoid log(0)
            additive_mask = torch.log(mask_attn.clamp(min=1e-6))  # [B, H, T, T]

            # Combine with causal mask
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )  # [T, T]
            combined_mask = additive_mask + causal_mask.unsqueeze(0).unsqueeze(0)

            # Expand combined_mask to match num_heads (repeat for each query head in group)
            # SDPA expects [B, num_heads, T, T]
            num_heads = config.num_attention_heads
            expanded_mask = combined_mask.repeat_interleave(q_per_kv, dim=1)  # [B, num_heads, T, T]

            # Inject mask via attention_mask kwarg
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = expanded_mask.to(hidden_states.dtype)
            else:
                kwargs["attention_mask"] = expanded_mask.to(hidden_states.dtype)

            # Zero the borrowed dimension in the hidden states so q_proj's borrowed neuron
            # doesn't influence attention scores during the actual attention computation.
            # We do this by creating a modified hidden_states where the dimension that maps
            # to the borrowed neuron output is zeroed.
            # However, this is complex because q_proj is a linear layer.
            # Instead, we zero it in the q_proj weight directly (it was already zeroed in phase 1).
            # The hook just needs to inject the mask.

            return args, kwargs

        return hook_fn

    # Register pre-hooks on each attention module
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        hooks.append(attn.register_forward_pre_hook(make_attn_pre_hook(layer_idx), with_kwargs=True))

    # Forward pass — hooks inject DMS masks into each attention layer
    outputs = model(input_ids=input_ids, use_cache=False)
    student_logits = outputs.logits

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Aggregate decisions
    decision_sum = sum(decision_sum_parts) if decision_sum_parts else torch.tensor(0.0, device=device)
    cap_loss = (
        torch.stack(cap_loss_parts).mean()
        if cap_loss_parts
        else torch.tensor(0.0, device=device)
    )
    cap_excess = (
        torch.stack(cap_excess_parts).mean().detach()
        if cap_excess_parts
        else torch.tensor(0.0, device=device)
    )
    cap_active_cells = (
        int((cap_loss_weights > 0).sum().item())
        if cap_loss_weights is not None
        else 0
    )
    cap_stats = {
        "cap_excess_mean": float(cap_excess.item()),
        "cap_active_cells": float(cap_active_cells),
    }
    total_elements = num_kv_heads * seq_len * batch * len(model.model.layers)

    return student_logits, decision_sum, total_elements, cap_loss, cap_stats


def load_cap_aware_profile(
    profile_path: str | None,
    *,
    profile_concurrency: int,
    hot_threshold: int,
    target_peak: int,
    physical_capacity: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
    """Build per-layer/head live-rate caps from a retained physical peak artifact."""
    if not profile_path:
        return None, None, {}

    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results", [])
    candidates = [
        row for row in rows
        if row.get("status") == "OK"
        and int(row.get("concurrency", -1)) == int(profile_concurrency)
        and row.get("dms_physical_peak_live_blocks_by_layer_head") is not None
    ]
    if not candidates:
        available = [
            row.get("concurrency")
            for row in rows
            if row.get("dms_physical_peak_live_blocks_by_layer_head") is not None
        ]
        raise ValueError(
            f"no cap-aware profile row for concurrency={profile_concurrency}; "
            f"available={available}"
        )

    matrix = torch.tensor(
        candidates[0]["dms_physical_peak_live_blocks_by_layer_head"],
        dtype=torch.float32,
    )
    hot_mask = matrix >= float(hot_threshold)
    cap_live_rate = min(1.0, max(0.0, float(target_peak) / float(physical_capacity)))
    cap_rates = torch.ones_like(matrix)
    cap_rates[hot_mask] = cap_live_rate
    weights = torch.zeros_like(matrix)
    if hot_mask.any():
        max_peak = float(matrix[hot_mask].max().item())
        weights[hot_mask] = matrix[hot_mask] / max(max_peak, 1.0)

    top = torch.nonzero(hot_mask, as_tuple=False)
    top_cells = []
    for layer, head in top.tolist():
        top_cells.append({
            "layer": int(layer),
            "head": int(head),
            "peak": int(matrix[layer, head].item()),
        })
    top_cells.sort(key=lambda item: item["peak"], reverse=True)

    summary = {
        "profile_path": str(path),
        "profile_concurrency": int(profile_concurrency),
        "hot_threshold": int(hot_threshold),
        "target_peak": int(target_peak),
        "physical_capacity": int(physical_capacity),
        "target_live_rate": cap_live_rate,
        "active_cells": int(hot_mask.sum().item()),
        "max_profile_peak": int(matrix.max().item()),
        "top_cells": top_cells[:16],
    }
    return cap_rates, weights, summary


# ---------------------------------------------------------------------------
# Neuron Zeroing (Phase 1)
# ---------------------------------------------------------------------------

def zero_borrowed_neuron_step(
    model: nn.Module,
    step: int,
    total_zeroing_steps: int,
):
    """
    Gradually zero the borrowed query dimension.

    At each step, multiply the last dimension of the first query head's q_proj weight
    (per GQA group) by (1 - step/total_zeroing_steps).

    After total_zeroing_steps, the dimension is fully zeroed.
    """
    factor = 1.0 - (step / total_zeroing_steps)
    config = model.config
    q_per_kv = config.num_attention_heads // config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    with torch.no_grad():
        for layer in model.model.layers:
            q_proj = layer.self_attn.q_proj
            # q_proj.weight shape: [num_heads * head_dim, hidden_size]
            # The borrowed dimension is the last dim of each GQA group's first query head.
            # In the weight matrix, head i's last dim is at row (i * head_dim + head_dim - 1).
            for group in range(config.num_key_value_heads):
                head_idx = group * q_per_kv  # first query head in this GQA group
                row_idx = head_idx * head_dim + head_dim - 1
                q_proj.weight.data[row_idx, :] *= factor
                if q_proj.bias is not None:
                    q_proj.bias.data[row_idx] *= factor


# ---------------------------------------------------------------------------
# Training Data
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Simple dataset of tokenized text chunks for DMS training."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = tokens.numel() // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len]


def load_training_data(
    tokenizer,
    dataset_name: str = "wikitext",
    seq_len: int = 4096,
    max_tokens: int = 4_000_000,
) -> TextDataset:
    """Load and tokenize training data."""
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
    elif dataset_name == "openr1math":
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train[:5000]")
        text = "\n\n".join(d.get("problem", "") + "\n" + d.get("solution", "") for d in ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    tokens = tokens[:max_tokens]
    print(f"Tokenized {tokens.numel()} tokens")
    return TextDataset(tokens, seq_len)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_dms(
    model_name: str,
    teacher_model_name: str | None = None,
    target_cr: int = 8,
    context_len: int = 4096,
    window_size: int = 256,
    zeroing_steps: int = 2000,
    steps_per_cr: int = 100,
    lr: float = 1e-5,
    tau_init: float = 1.0,
    tau_final: float = 0.1,
    alpha_scale: float = 100.0,
    alpha_offset: float = 5.0,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    compression_weight: float = 1.0,
    compression_mode: str = "relu",
    device: str = "cuda:1",
    output_dir: str = "results/dms",
    dataset: str = "wikitext",
    log_every: int = 10,
    save_every: int = 500,
    cap_aware_profile: str | None = None,
    cap_aware_profile_concurrency: int = 8,
    cap_aware_hot_threshold: int = 20_000,
    cap_aware_target_peak: int = 20_000,
    cap_aware_physical_capacity: int = 25_088,
    cap_aware_weight: float = 0.0,
):
    """
    Full DMS training: Phase 1 (neuron zeroing) + Phase 2 (DMS retrofitting).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== DMS Training ===")
    print(f"Model: {model_name}")
    print(f"Target CR: {target_cr}")
    print(f"Context: {context_len}")
    print(f"Window: {window_size}")
    print(f"Device: {device}")
    print(f"Zeroing steps: {zeroing_steps}")
    print(f"Retrofitting steps: {steps_per_cr * (target_cr - 1)}")
    print(f"Compression weight: {compression_weight}")
    print(f"Compression mode: {compression_mode}")
    print(f"Cap-aware weight: {cap_aware_weight}")
    total_steps = zeroing_steps + steps_per_cr * (target_cr - 1)
    print(f"Total steps: {total_steps}")

    # Load teacher (frozen) and student (trainable) models.  The teacher can
    # stay on the original model when continuing from a DMS checkpoint.
    teacher_model_name = teacher_model_name or model_name
    print("\nLoading teacher model (frozen)...")
    print(f"Teacher: {teacher_model_name}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("Loading student model (trainable)...")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    student.train()

    # Only train q_proj weights (the borrowed neuron lives there)
    # and optionally the full model for adaptation
    for name, param in student.named_parameters():
        param.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    train_dataset = load_training_data(tokenizer, dataset_name=dataset, seq_len=context_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    config = student.config
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    q_per_kv = config.num_attention_heads // num_kv_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    cap_live_rates, cap_loss_weights, cap_summary = load_cap_aware_profile(
        cap_aware_profile,
        profile_concurrency=cap_aware_profile_concurrency,
        hot_threshold=cap_aware_hot_threshold,
        target_peak=cap_aware_target_peak,
        physical_capacity=cap_aware_physical_capacity,
    )
    if cap_live_rates is not None:
        if tuple(cap_live_rates.shape) != (num_layers, num_kv_heads):
            raise ValueError(
                "cap-aware profile shape mismatch: "
                f"got {tuple(cap_live_rates.shape)}, expected {(num_layers, num_kv_heads)}"
            )
        cap_live_rates = cap_live_rates.to(device=device)
        cap_loss_weights = cap_loss_weights.to(device=device)
        print(
            "Cap-aware profile: "
            f"{cap_summary['active_cells']} hot cells, "
            f"target_live_rate={cap_summary['target_live_rate']:.4f}, "
            f"max_peak={cap_summary['max_profile_peak']}"
        )
        for cell in cap_summary["top_cells"][:8]:
            print(
                "  hot cell "
                f"L{cell['layer']} H{cell['head']} peak={cell['peak']}"
            )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Training state
    step = 0
    data_iter = iter(train_loader)
    log_data = []
    t0 = time.time()

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        return batch.to(device)

    # -----------------------------------------------------------------------
    # Phase 1: Neuron Zeroing
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Neuron Zeroing ({zeroing_steps} steps)")
    print(f"{'='*60}")

    for step in range(1, zeroing_steps + 1):
        input_ids = get_batch()

        # Gradually zero the borrowed dimension
        zero_borrowed_neuron_step(student, step, zeroing_steps)

        # Forward through student
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            student_out = student(input_ids=input_ids, use_cache=False)
            student_logits = student_out.logits

            # Teacher forward
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, use_cache=False)
                teacher_logits = teacher_out.logits

            # KL distillation loss (student should match teacher)
            loss = F.kl_div(
                F.log_softmax(student_logits[:, :-1].float(), dim=-1),
                F.softmax(teacher_logits[:, :-1].float(), dim=-1),
                reduction="batchmean",
            )

        loss.backward()

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            elapsed = time.time() - t0
            entry = {"step": step, "phase": 1, "loss": loss.item(), "elapsed_sec": elapsed}
            log_data.append(entry)
            print(f"  Step {step}/{zeroing_steps} | KD Loss: {loss.item():.4f} | {elapsed:.0f}s")

    # Verify zeroing is complete
    with torch.no_grad():
        sample_layer = student.model.layers[0].self_attn.q_proj
        row_idx = (head_dim - 1)  # first group's first head, last dim
        max_val = sample_layer.weight.data[row_idx].abs().max().item()
        print(f"Zeroing check — layer 0, borrowed dim max weight: {max_val:.2e}")

    # -----------------------------------------------------------------------
    # Phase 2: DMS Retrofitting (with soft attention masking)
    # -----------------------------------------------------------------------
    retrofit_steps = steps_per_cr * (target_cr - 1)
    print(f"\n{'='*60}")
    print(f"Phase 2: DMS Retrofitting ({retrofit_steps} steps, CR 1→{target_cr})")
    print(f"{'='*60}")

    for phase2_step in range(1, retrofit_steps + 1):
        step = zeroing_steps + phase2_step
        input_ids = get_batch()

        # Current target CR (linearly annealed from 1 to target_cr)
        progress = phase2_step / retrofit_steps
        current_cr = 1.0 + (target_cr - 1.0) * progress
        target_eviction_rate = 1.0 - 1.0 / current_cr  # fraction of tokens to evict

        # Current tau (anneal for sharper decisions)
        tau = tau_init + (tau_final - tau_init) * progress

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Student forward WITH DMS soft attention masking.
            # Decisions are extracted from the borrowed neuron and used to build
            # soft attention masks injected into each layer. This means KD loss
            # gradients flow through attention → mask → Gumbel-sigmoid → q_proj,
            # giving the model a learning signal for which tokens to evict.
            student_logits, decision_sum, total_decision_elements, cap_loss, cap_stats = dms_forward_with_masking(
                student, input_ids,
                alpha_scale=alpha_scale,
                alpha_offset=alpha_offset,
                tau=tau,
                window_size=window_size,
                q_per_kv=q_per_kv,
                hard=False,
                cap_live_rates=cap_live_rates,
                cap_loss_weights=cap_loss_weights,
            )

            # Teacher forward (always full attention, no masking)
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, use_cache=False)
                teacher_logits = teacher_out.logits

            # Loss 1: KL distillation
            kd_loss = F.kl_div(
                F.log_softmax(student_logits[:, :-1].float(), dim=-1),
                F.softmax(teacher_logits[:, :-1].float(), dim=-1),
                reduction="batchmean",
            )

            # Loss 2: Compression auxiliary
            target_evictions = target_eviction_rate * total_decision_elements
            actual_eviction_rate = decision_sum / max(1, total_decision_elements)

            if compression_mode == "l2":
                # L2 penalty: (target_rate - actual_rate)^2
                # Pushes harder when the gap is large
                compression_loss = (target_eviction_rate - actual_eviction_rate) ** 2
            else:
                # One-sided ReLU (paper default): max(target - actual, 0)
                compression_loss = F.relu(target_eviction_rate - actual_eviction_rate)

            weighted_cap_loss = cap_aware_weight * cap_loss
            loss = kd_loss + compression_weight * compression_loss + weighted_cap_loss

        loss.backward()

        if phase2_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if phase2_step % log_every == 0:
            elapsed = time.time() - t0
            actual_rate = actual_eviction_rate.item()

            # Log gradient norms for the borrowed neuron to verify gradient flow
            grad_norm_borrowed = 0.0
            n_layers_with_grad = 0
            for layer in student.model.layers:
                q_proj = layer.self_attn.q_proj
                if q_proj.weight.grad is not None:
                    # Borrowed dim is at row (group_idx * q_per_kv * head_dim + head_dim - 1)
                    row_idx = head_dim - 1  # first group
                    grad_norm_borrowed += q_proj.weight.grad[row_idx].norm().item()
                    n_layers_with_grad += 1
            avg_grad_norm = grad_norm_borrowed / max(1, n_layers_with_grad)

            entry = {
                "step": step, "phase": 2,
                "kd_loss": kd_loss.item(),
                "compression_loss": compression_loss.item(),
                "weighted_comp_loss": (compression_weight * compression_loss).item(),
                "cap_aware_loss": cap_loss.item(),
                "weighted_cap_aware_loss": weighted_cap_loss.item(),
                "cap_aware_excess_mean": cap_stats["cap_excess_mean"],
                "cap_aware_active_cells": cap_stats["cap_active_cells"],
                "loss": loss.item(),
                "target_cr": current_cr,
                "target_eviction_rate": target_eviction_rate,
                "actual_eviction_rate": actual_rate,
                "tau": tau,
                "borrowed_neuron_grad_norm": avg_grad_norm,
                "elapsed_sec": elapsed,
            }
            log_data.append(entry)
            print(
                f"  Step {step}/{zeroing_steps + retrofit_steps} | "
                f"CR {current_cr:.1f} | "
                f"KD {kd_loss.item():.4f} | "
                f"Comp {compression_loss.item():.4f}×{compression_weight:.0f} | "
                f"Cap {cap_loss.item():.4f}×{cap_aware_weight:.0f} | "
                f"Evict {actual_rate:.3f}/{target_eviction_rate:.3f} | "
                f"tau {tau:.2f} | "
                f"∇borrow {avg_grad_norm:.2e} | "
                f"{elapsed:.0f}s"
            )

        if phase2_step % save_every == 0 or phase2_step == retrofit_steps:
            ckpt_cr = int(current_cr)
            ckpt_path = output_path / f"checkpoint-cr{ckpt_cr}-step{step}"
            print(f"  Saving checkpoint: {ckpt_path}")
            student.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # Save final checkpoint and training log
    final_path = output_path / "final"
    print(f"\nSaving final checkpoint: {final_path}")
    student.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    log_path = output_path / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "model": model_name,
                "teacher_model": teacher_model_name,
                "target_cr": target_cr,
                "context_len": context_len,
                "window_size": window_size,
                "zeroing_steps": zeroing_steps,
                "steps_per_cr": steps_per_cr,
                "lr": lr,
                "alpha_scale": alpha_scale,
                "alpha_offset": alpha_offset,
                "compression_weight": compression_weight,
                "compression_mode": compression_mode,
                "cap_aware_weight": cap_aware_weight,
                "cap_aware_profile": cap_aware_profile,
                "cap_aware_profile_summary": cap_summary,
            },
            "log": log_data,
        }, f, indent=2)
    print(f"Training log: {log_path}")
    print(f"Total time: {time.time() - t0:.0f}s")

    return student


def main():
    parser = argparse.ArgumentParser(description="Train DMS for KV cache token eviction")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Optional frozen teacher model; useful when continuing from a DMS checkpoint.",
    )
    parser.add_argument("--target-cr", type=int, default=8, help="Target compression ratio")
    parser.add_argument("--context-len", type=int, default=4096, help="Training context length")
    parser.add_argument("--window-size", type=int, default=256, help="DMS sliding window size")
    parser.add_argument("--zeroing-steps", type=int, default=2000, help="Phase 1 neuron zeroing steps")
    parser.add_argument("--steps-per-cr", type=int, default=100, help="Phase 2 steps per CR unit")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--compression-weight", type=float, default=1.0, help="Compression loss multiplier")
    parser.add_argument("--compression-mode", type=str, default="relu", choices=["relu", "l2"], help="Compression loss type")
    parser.add_argument("--cap-aware-profile", type=str, default=None, help="Physical peak artifact used to build per-head live-rate caps")
    parser.add_argument("--cap-aware-profile-concurrency", type=int, default=8, help="Profile row concurrency to use for cap-aware training")
    parser.add_argument("--cap-aware-hot-threshold", type=int, default=20_000, help="Only profile cells at or above this peak receive cap loss")
    parser.add_argument("--cap-aware-target-peak", type=int, default=20_000, help="Target per-layer/head peak used to derive live-rate cap")
    parser.add_argument("--cap-aware-physical-capacity", type=int, default=25_088, help="Physical cap used to convert target peak to live-rate cap")
    parser.add_argument("--cap-aware-weight", type=float, default=0.0, help="Per-head cap loss multiplier")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Training dataset")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    args = parser.parse_args()

    train_dms(
        model_name=args.model,
        teacher_model_name=args.teacher_model,
        target_cr=args.target_cr,
        context_len=args.context_len,
        window_size=args.window_size,
        zeroing_steps=args.zeroing_steps,
        steps_per_cr=args.steps_per_cr,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        compression_weight=args.compression_weight,
        compression_mode=args.compression_mode,
        device=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
        log_every=args.log_every,
        save_every=args.save_every,
        cap_aware_profile=args.cap_aware_profile,
        cap_aware_profile_concurrency=args.cap_aware_profile_concurrency,
        cap_aware_hot_threshold=args.cap_aware_hot_threshold,
        cap_aware_target_peak=args.cap_aware_target_peak,
        cap_aware_physical_capacity=args.cap_aware_physical_capacity,
        cap_aware_weight=args.cap_aware_weight,
    )


if __name__ == "__main__":
    main()
