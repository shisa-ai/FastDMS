"""Build document-disjoint Nanbeige DMS streams from local Shisa and Sansho data."""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

SOURCES = {
    'shisa-v2.1': '/home/lhl/shisa-v2/train/v2.1/sft.shisa-v2.1.jsonl',
    **{name: f'/home/lhl/sansho/outputs/data/cache/tx4-quality2-5b/{name}/docs-*.jsonl'
       for name in ['finewiki-en', 'finewiki-ja', 'stack-edu-python-materialized', 'pg19', 'swallow-math-v2-textbook']},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='primitive-ai/Nanbeige4.3-33B-FP8')
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--exclude-manifest', action='append', default=[])
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--train-per-source', type=int, default=8)
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quotas = {'train': args.train_per_source, 'validation': 0 if args.train_only else 1, 'final': 0 if args.train_only else 1}
    records, seen = [], set()
    for file in args.exclude_manifest:
        old = json.loads(Path(file).read_text())
        seen.update(d['sha256'] for r in old['records'] for d in r['documents'])
    for source, pattern in SOURCES.items():
        buffers = {s: [] for s in quotas}; documents = {s: [] for s in quotas}
        counts = {s: 0 for s in quotas}
        for file in sorted(glob.glob(pattern)):
            with open(file) as handle:
                for line_index, line in enumerate(handle):
                    row = json.loads(line)
                    text = row.get('text')
                    if text is None:
                        text = tokenizer.apply_chat_template(row['conversations'], tokenize=False, add_generation_prompt=False)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in seen: continue
                    seen.add(digest)
                    bucket = int(digest[:8], 16) % 20
                    split = 'final' if bucket == 0 else ('validation' if bucket == 1 else 'train')
                    if counts[split] >= quotas[split]: continue
                    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                    buffers[split].extend(ids)
                    documents[split].append(dict(path=file, line=line_index, sha256=digest,
                                                 source_doc_index=row.get('source_doc_index')))
                    # Discard the document tail after filling a stream: a document
                    # never crosses streams or splits, and final sources stay sealed.
                    if len(buffers[split]) >= args.context:
                        name = f'{source}-{split}-{counts[split]:03d}.pt'
                        torch.save(torch.tensor(buffers[split][:args.context], dtype=torch.long), root / name)
                        records.append(dict(path=name, source=source, split=split, tokens=args.context,
                                            documents=documents[split]))
                        buffers[split] = []; documents[split] = []; counts[split] += 1
                    if all(counts[s] >= quotas[s] for s in quotas): break
            if all(counts[s] >= quotas[s] for s in quotas): break
        if counts != quotas: raise RuntimeError(f'insufficient real data for {source}: {counts}')
        print(source, counts, flush=True)
    manifest = dict(model=args.model, context=args.context, split_rule='sha256(text) modulo 20: 0 final, 1 validation, remainder train',
                    records=records, training_tokens=sum(r['tokens'] for r in records if r['split']=='train'))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()
