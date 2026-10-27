"""Capture real FP8 serving activations and long-context future-use labels."""
import argparse
import json
from pathlib import Path
import torch
from fastdms import LLM, SamplingParams
from fastdms.engine.sequence import Sequence
from fastdms.utils.context import reset_context, get_context
from training.sidecar import future_attention_mass, eviction_labels, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--rows', type=int, default=1024)
    p.add_argument('--query-stride', type=int, default=16)
    args = p.parse_args()
    data = Path(args.data); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/'manifest.json').read_text())
    context = manifest['context']
    llm = LLM(args.model, enforce_eager=True, max_model_len=context, max_num_batched_tokens=context,
              max_num_seqs=1, num_page_override=context//256, gpu_memory_utilization=0.85)
    runner = llm.model_runner
    rows = []
    for index, record in enumerate(manifest['records']):
        if record['split'] == 'final' or index % args.world_size != args.rank: continue
        target = output / record['path'].replace('.pt', '')
        target.mkdir(exist_ok=True)
        handles = []; hidden = {}; current = [None]
        def input_hook(layer):
            def hook(module, inputs):
                hidden[layer] = inputs[0].detach()
                current[0] = layer
            return hook
        def rotary_hook():
            def hook(module, inputs, outputs):
                layer = current[0]
                q, k = outputs
                with torch.no_grad():
                    mass = future_attention_mass(q.transpose(0,1)[None], k.transpose(0,1)[None],
                                                 8192, query_chunk=256, query_stride=args.query_stride)
                    labels, eligible = eviction_labels(mass, 8192, 2)
                    generator = torch.Generator().manual_seed(20261026 + index * 52 + layer)
                    selected = torch.randperm(eligible, generator=generator)[:args.rows].sort().values.to(q.device)
                    payload = dict(hidden=hidden.pop(layer)[selected].cpu().bfloat16(),
                                   labels=labels[0,selected].cpu(), positions=selected.cpu())
                    file = target/f'layer-{layer:02d}.pt'
                    torch.save(payload, file)
                    rows.append(dict(path=str(file.relative_to(output)), layer=layer, source=record['source'],
                                     split=record['split'], sequence=record['path'], sha256=sha256_file(file)))
            return hook
        for layer, block in enumerate(runner.model.model.layers):
            handles.append(block.self_attn.qkv_proj.register_forward_pre_hook(input_hook(layer)))
        # Rotary module is shared across layers; register one dispatching hook.
        handles.append(runner.model.model.layers[0].self_attn.rotary_emb.register_forward_hook(rotary_hook()))
        ids = torch.load(data/record['path'], weights_only=True).tolist()
        seq = Sequence(ids, SamplingParams(max_tokens=1));seq.num_scheduled_tokens=len(ids)
        try:
            inputs, positions = runner.prepare_prefill([seq])
            # Full-prompt capture needs no persistent decode KV cache.
            get_context().slot_mapping = None
            with torch.inference_mode(): runner.run_model(inputs, positions, True)
        finally:
            for h in handles: h.remove()
            reset_context()
        print(f'rank {args.rank} captured {record["path"]}', flush=True)
        (output/f'manifest-rank{args.rank}.json').write_text(json.dumps(dict(
            model=args.model, data_sha256=sha256_file(data/'manifest.json'), window=8192,
            target_cr=2, query_stride=args.query_stride, sampled_rows=args.rows, records=rows), indent=2)+'\n')


if __name__ == '__main__': main()
