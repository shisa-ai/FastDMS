"""Train frozen Llama-family DMS sidecars with one replica per GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .sidecar import LinearSidecar, eviction_labels, export_sidecar, future_attention_mass, sha256_file


def train_batch(base, sidecar, tokens, *, window_size, target_cr, query_chunk=128, loss_scale=1.0):
    """Capture normalized inputs and post-RoPE Q/K without a base-model graph.

    Backpropagate each independent head immediately. Only head parameters receive
    gradients; no full-model activations or vocabulary logits are retained.
    """
    losses = []
    handles = []

    def make_hook(index):
        def capture(attention, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            batch, length, _ = hidden.shape
            dim = attention.head_dim
            q = attention.q_proj(hidden).view(batch, length, -1, dim).transpose(1, 2)
            k = attention.k_proj(hidden).view(batch, length, -1, dim).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            mass = future_attention_mass(q, k, window_size, query_chunk)
            labels, eligible = eviction_labels(mass, window_size, target_cr)
            if not eligible:
                raise ValueError("context must contain tokens outside the protected window")
            with torch.enable_grad():
                logits = sidecar(hidden[:, :eligible].detach(), index)
                loss = F.binary_cross_entropy_with_logits(logits, labels[:, :eligible])
                (loss * loss_scale / len(base.model.layers)).backward()
            losses.append(loss.detach())
        return capture

    try:
        for index, layer in enumerate(base.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make_hook(index), with_kwargs=True))
        with torch.no_grad():
            base.model(input_ids=tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(losses).mean()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Nanbeige/Nanbeige4.3-33B")
    p.add_argument("--tokens", required=True, help="Local .pt int64 tensor [examples, context], without padding")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--target-cr", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--local-files-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if (args.steps <= 0 or args.grad_accum <= 0 or args.lr <= 0 or args.seed < 0
            or args.window_size <= 0 or args.context_len <= args.window_size + 1
            or args.target_cr < 2 or args.query_chunk <= 0):
        raise ValueError("require positive steps, accumulation, lr, window and query chunk; context > window + 1; CR >= 2; seed >= 0")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        torch.manual_seed(args.seed)
        config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
        if config.model_type != "llama":
            raise ValueError("this trainer supports Llama architecture, including Nanbeige4.3-33B")
        tokens = torch.load(args.tokens, map_location="cpu", weights_only=True)
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tokens.ndim != 2
                or tokens.shape[0] < world or tokens.shape[1] < args.context_len
                or tokens.min() < 0 or tokens.max() >= config.vocab_size):
            raise ValueError("tokens must be int64 [examples >= world size, context >= context-len] within model vocabulary")
        # Bind the source model to its config and exact HF shard identities.
        from huggingface_hub import snapshot_download
        root = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(
            args.model, local_files_only=args.local_files_only))
        manifest = {"model": args.model, "config_sha256": sha256_file(root / "config.json"),
                    "weights": {}}
        if rank == 0:
            for path in sorted(root.glob("*.safetensors")):
                manifest["weights"][path.name] = sha256_file(path)
            if not manifest["weights"]:
                raise ValueError("model must contain safetensors weights")
        objects = [manifest]
        if world > 1:
            dist.broadcast_object_list(objects, src=0, device=device)
        manifest = objects[0]
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        print(f"rank={rank} loading frozen model on {device}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(root), dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": str(device)}, local_files_only=True)
        base.eval().requires_grad_(False)
        sidecar = LinearSidecar(config.num_hidden_layers, config.num_key_value_heads,
                                config.hidden_size).to(device)
        optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.lr, weight_decay=0)
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(tokens), generator=generator)
        history = []
        started = time.monotonic()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for micro in range(args.grad_accum):
                sample = ((step * args.grad_accum + micro) * world + rank) % len(tokens)
                batch = tokens[order[sample]:order[sample] + 1, :args.context_len].to(device)
                loss += train_batch(base, sidecar, batch, window_size=args.window_size,
                                    target_cr=args.target_cr, query_chunk=args.query_chunk,
                                    loss_scale=1 / args.grad_accum) / args.grad_accum
            if world > 1:
                for parameter in sidecar.parameters():
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
                dist.all_reduce(loss)
                loss.div_(world)
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in sidecar.parameters()):
                raise FloatingPointError("non-finite sidecar loss or gradient")
            norm = torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
            optimizer.step()
            entry = dict(step=step + 1, loss=loss.item(), grad_norm=norm.item(),
                         elapsed_seconds=time.monotonic() - started)
            history.append(entry)
            if rank == 0:
                print(json.dumps(entry), flush=True)
        if rank == 0:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            data_manifest = dict(tokens_sha256=sha256_file(args.tokens), shape=list(tokens.shape),
                                 context_len=args.context_len, seed=args.seed)
            data_bytes = (json.dumps(data_manifest, sort_keys=True, indent=2) + "\n").encode()
            (output / "data_manifest.json").write_bytes(data_bytes)
            (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            summary = dict(status="trained_candidate", quality_qualified=False, args=vars(args),
                           world_size=world, history=history,
                           peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                           source_sha256={name: sha256_file(Path(__file__).parent / name)
                                          for name in ("sidecar.py", "train_sidecar.py")})
            export_sidecar(sidecar, config, output, artifact_fingerprint=fingerprint,
                           data_manifest_sha256=hashlib.sha256(data_bytes).hexdigest(),
                           trainer_commit=commit, seed=args.seed, window_size=args.window_size,
                           target_cr=args.target_cr, summary=summary)
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
