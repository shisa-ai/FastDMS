"""Paired teacher-forced KV quality checks; dense BF16 KV is the reference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from fastdms import LLM, SamplingParams


PROSE = [
    ("science", "A sealed glass container holds water and air. When it is warmed gently, evaporation increases, but no matter can leave the container. ",
     "The total mass remains constant because the system is closed. Water changes phase without creating or destroying matter."),
    ("code", "A Python function receives a list of integers and must return the unique values in ascending order. The input may contain duplicates and negative numbers. ",
     "def unique_sorted(values):\n    return sorted(set(values))\n\nThis creates a set to remove duplicates and sorts its elements."),
    ("japanese", "図書館では、本を借りるために利用者カードが必要です。貸出期間は二週間で、予約がなければ延長できます。",
     "返却期限を確認し、読み終わった本はカウンターに返してください。延長を希望する場合は、期限前に手続きを行います。"),
]


def cases(tokenizer):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)
    for name, prefix, answer in PROSE:
        ids = encode(prefix)
        yield name, (ids * math.ceil(512 / len(ids)))[:512], encode(answer)[:32]
    filler = encode("Archive entry: routine observations were recorded. This entry contains no access code.\n")
    needle = encode("\nIMPORTANT: The access code for project Cedar is 739281.\n")
    suffix = encode("\nQuestion: What is the access code for project Cedar? Answer with the six digits only.\nAnswer:")
    for length in (32768, 131072):
        for fraction in (0.1, 0.8):
            room = length - len(needle) - len(suffix)
            background = (filler * math.ceil(room / len(filler)))[:room]
            split = int(room * fraction)
            prompt = background[:split] + needle + background[split:] + suffix
            yield f"retrieval-{length}-{fraction}", prompt, encode("739281")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--metadata')
    p.add_argument('--kv-dtype', default='bfloat16')
    p.add_argument('--reference', required=True, help='Directory written by dense BF16 KV run')
    p.add_argument('--output', required=True)
    p.add_argument('--data-manifest', help='Real-data manifest; evaluate heldout stream continuations instead of synthetic cases')
    p.add_argument('--split', choices=['validation', 'final'], default='validation')
    p.add_argument('--write-reference', action='store_true')
    args = p.parse_args()
    if args.write_reference and (args.metadata or args.kv_dtype != 'bfloat16'):
        p.error('reference must use dense BF16 KV')
    os.environ['FASTDMS_COMPACT_GREEDY_FAST_LOOP'] = '0'
    os.environ['FASTDMS_COMPACT_DECODE_CUDAGRAPH'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = dict(enforce_eager=True, max_model_len=131136, max_num_batched_tokens=131072,
                  max_num_seqs=1, gpu_memory_utilization=0.94, kv_cache_dtype=args.kv_dtype,
                  num_page_override=513)
    if args.metadata:
        kwargs.update(compact_kv_enabled=True, compact_kv_retention_mode='dms',
                      compact_kv_capacity_tokens=8 * 131136, dms_metadata_path=args.metadata)
    llm = LLM(args.model, **kwargs)
    reference = Path(args.reference)
    reference.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    evaluation_cases = cases(tokenizer)
    if args.data_manifest:
        manifest_path = Path(args.data_manifest)
        manifest = json.loads(manifest_path.read_text())
        def real_cases():
            for record in manifest['records']:
                if record['split'] != args.split:
                    continue
                ids = torch.load(manifest_path.parent / record['path'], weights_only=True).tolist()
                yield record['path'].removesuffix('.pt'), ids[:-32], ids[-32:]
        evaluation_cases = real_cases()
    for name, prompt, targets in evaluation_cases:
        print(f'Starting {name}: {len(prompt)} prompt tokens, {len(targets)} scored tokens', flush=True)
        identity = hashlib.sha256(json.dumps([prompt, targets]).encode()).hexdigest()
        captured = []
        def force_tokens(logits, temperatures):
            captured.append(logits.detach().float().cpu()[0])
            return torch.tensor([targets[len(captured) - 1]], device=logits.device)
        llm.model_runner.sampler.forward = force_tokens
        llm.scheduler.block_manager.clear_prefix_cache()
        llm.generate([prompt], SamplingParams(temperature=1.0, max_tokens=len(targets), ignore_eos=True), use_tqdm=False)
        actual = torch.stack(captured)
        if not torch.isfinite(actual).all():
            raise FloatingPointError(f'non-finite logits for {name}')
        file = reference / (name + '.pt')
        if args.write_reference:
            torch.save(dict(logits=actual, identity=identity), file)
        saved = torch.load(file, weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('reference prompt/target identity mismatch')
        ref = saved['logits']
        logp, ref_logp = actual.log_softmax(-1), ref.log_softmax(-1)
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)
        rows = torch.arange(len(targets))
        nll = -logp[rows, torch.tensor(targets)]
        ref_nll = -ref_logp[rows, torch.tensor(targets)]
        # Prefill is dense in all modes. Decode-only metrics expose retained KV differences.
        result = dict(case=name, prompt_tokens=len(prompt), scored_tokens=len(targets), identity=identity,
                      mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                      decode_mean_kl=float(kl[1:].mean()),
                      top1_agreement=float((actual.argmax(-1) == ref.argmax(-1)).float().mean()),
                      decode_top1_agreement=float((actual[1:].argmax(-1) == ref[1:].argmax(-1)).float().mean()),
                      target_nll=float(nll.mean()), reference_target_nll=float(ref_nll.mean()),
                      target_token_accuracy=float((actual.argmax(-1) == torch.tensor(targets)).float().mean()),
                      teacher_forced_predictions=tokenizer.decode(actual.argmax(-1).tolist()),
                      expected=tokenizer.decode(targets))
        results.append(result)
        output.write_text(json.dumps(dict(model=args.model, kv_dtype=args.kv_dtype, metadata=args.metadata,
                                         protocol='teacher_forced_fixed_continuation', results=results), indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
