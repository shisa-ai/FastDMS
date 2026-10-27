"""Fit long-context sidecar captures with train-only calibration and heldout BCE."""
import argparse
import json
import random
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from training.sidecar import LinearSidecar, export_sidecar, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', required=True); p.add_argument('--output', required=True)
    p.add_argument('--model', required=True); p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--initial-training-state', help='Continue optimizer and weights while starting a new data epoch')
    p.add_argument('--initial-checkpoint', help='Start fitting from uncalibrated sidecar weights')
    p.add_argument('--export-checkpoint', help='Calibrate/export an existing epoch checkpoint without fitting')
    p.add_argument('--resume', action='store_true'); p.add_argument('--seed', type=int, default=20261026)
    args = p.parse_args()
    root, output = Path(args.labels), Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    manifests = [json.loads(p.read_text()) for p in sorted(root.glob('manifest-rank*.json'))]
    data_manifest = root.parent/'data/manifest.json'
    data = json.loads(data_manifest.read_text())
    expected = {(r['path'], layer) for r in data['records'] if r['split'] != 'final' for layer in range(52)}
    records = [r for m in manifests for r in m['records']]
    actual = {(r['sequence'],r['layer']) for r in records}
    if actual != expected or len(actual) != len(records):
        raise ValueError(f'incomplete or duplicate capture: {len(actual)}/{len(expected)} shards')
    for r in records:
        if sha256_file(root/r['path']) != r['sha256']: raise ValueError('capture hash mismatch')
    config = AutoConfig.from_pretrained(args.model)
    model = LinearSidecar(52, 8, 7168).cuda()
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(args.initial_checkpoint, weights_only=True))
    layer_parameters = [(torch.nn.Parameter(model.weight[i].detach()),
                         torch.nn.Parameter(model.bias[i].detach())) for i in range(52)]
    optimizers = [torch.optim.AdamW(params, lr=0.001, weight_decay=0) for params in layer_parameters]
    checkpoint = output/'resume.pt'; history = []; start = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state['model'])
        for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
        history = state['history']; start = state['epoch']
    if args.initial_training_state:
        state = torch.load(args.initial_training_state, weights_only=True)
        model.load_state_dict(state['model'])
        for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
    if args.export_checkpoint:
        model.load_state_dict(torch.load(args.export_checkpoint, weights_only=True))
        start = args.epochs
    # Layer-wise residency: roughly 750 MiB per layer, rather than all 40 GiB.
    for epoch in range(start, args.epochs):
        totals = {split: [0.0, 0, 0] for split in ('train','validation')}
        for layer in range(52):
            optimizer = optimizers[layer]
            params = layer_parameters[layer]
            shards = {split: [] for split in totals}
            for r in records:
                if r['layer'] == layer: shards[r['split']].append(torch.load(root/r['path'], weights_only=True))
            for split in ('train','validation'):
                x = torch.cat([s['hidden'] for s in shards[split]]).cuda().float()
                y = torch.cat([s['labels'] for s in shards[split]]).cuda()
                gen = torch.Generator().manual_seed(args.seed + epoch*52+layer)
                order = torch.randperm(len(x), generator=gen) if split=='train' else torch.arange(len(x))
                for ids in order.split(512):
                    ids = ids.cuda()
                    with torch.set_grad_enabled(split=='train'):
                        logits = F.linear(x[ids], *params)
                        bce = F.binary_cross_entropy_with_logits(logits,y[ids])
                        budget = (logits.sigmoid().mean(0)-y[ids].mean(0)).square().mean()
                        if split=='train':
                            optimizer.zero_grad(set_to_none=True)
                            (bce + 0.1*budget).backward()
                            torch.nn.utils.clip_grad_norm_(params,1)
                            optimizer.step()
                    totals[split][0] += bce.item()*y[ids].numel()
                    totals[split][1] += int(((logits>0)==y[ids].bool()).sum())
                    totals[split][2] += y[ids].numel()
                del x,y
        row = dict(epoch=epoch+1, **{s: dict(bce=v[0]/v[2],accuracy=v[1]/v[2],decisions=v[2]) for s,v in totals.items()})
        history.append(row); print(json.dumps(row),flush=True)
        torch.save(dict(epoch=epoch+1, model=model.state_dict(), optimizers=[optimizer.state_dict() for optimizer in optimizers],history=history),checkpoint)
        torch.save(model.state_dict(),output/f'epoch-{epoch+1:02d}.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    # Train-only per-head median calibration, evaluated in exported BF16 arithmetic.
    with torch.no_grad():
        model.weight.copy_(model.weight.bfloat16().float());model.bias.copy_(model.bias.bfloat16().float())
        for layer in range(52):
            scores=[]
            for r in records:
                if r['layer']==layer and r['split']=='train':
                    s=torch.load(root/r['path'],weights_only=True)
                    scores.append(model(s['hidden'].cuda(),layer).cpu())
            threshold=torch.cat(scores).median(dim=0).values.cuda()
            model.bias[layer].sub_(threshold)
    data_manifest = root.parent/'data/manifest.json'
    model_manifest = dict(config=sha256_file(Path(args.model)/'config.json'),weights={})
    for file in sorted(Path(args.model).glob('*.safetensors')): model_manifest['weights'][file.name]=sha256_file(file)
    import hashlib
    artifact=hashlib.sha256(json.dumps(model_manifest,sort_keys=True).encode()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    metadata=export_sidecar(model,config,output/'final',artifact_fingerprint=artifact,
        data_manifest_sha256=sha256_file(data_manifest),trainer_commit=commit,seed=args.seed,
        window_size=8192,target_cr=2,summary=dict(history=history,quality_qualified=False,
        capture_manifests=[sha256_file(f) for f in sorted(root.glob('manifest-rank*.json'))],
        query_stride=manifests[0]['query_stride'],training_tokens=data['training_tokens'],source_hash=sha256_file(__file__)))
    metadata['prefill_selection_mode']='exact_budget'
    (output/'final/dms_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__': main()
