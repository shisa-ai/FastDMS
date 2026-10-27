"""Run ten continuation epochs with 24 fresh and 24 replay 64K streams per epoch."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(module, *args, gpu=None):
    env=os.environ.copy()
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    subprocess.run([sys.executable,'-m',module,*map(str,args)],env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--gpu',default='0')
    args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=Path('artifacts/nanbeige-real64')
    original=json.loads((base/'data/manifest.json').read_text())
    capture=json.loads((base/'labels/manifest-rank0.json').read_text())
    excludes=[Path('artifacts/nanbeige-real/data/manifest.json'),base/'data/manifest.json']
    previous=None
    for epoch in range(1,args.epochs+1):
        stage=root/f'epoch-{epoch:02d}'
        stage.mkdir(exist_ok=True)
        completion=stage/'complete.json'
        if completion.exists():
            excludes.append(stage/'fresh-data/manifest.json');previous=stage/'training/resume.pt';continue
        command=['--output',stage/'fresh-data','--context','65536','--train-per-source','4','--train-only']
        for exclude in excludes: command+=['--exclude-manifest',exclude]
        run('training.prepare_real_data',*command)
        fresh=json.loads((stage/'fresh-data/manifest.json').read_text())
        run('training.capture_real_sidecar','--data',stage/'fresh-data','--output',stage/'fresh-labels',
            '--model',args.model,'--rank','0','--world-size','1',gpu=args.gpu)
        fresh_capture=json.loads((stage/'fresh-labels/manifest-rank0.json').read_text())
        # Alternate halves of the original training corpus for replay. Validation is fixed.
        selected=[]; per_source={}
        for record in original['records']:
            if record['split']=='final':continue
            if record['split']=='validation':selected.append(record);continue
            index=per_source.get(record['source'],0);per_source[record['source']]=index+1
            if index//4 == (epoch-1)%2:selected.append(record)
        names={r['path'] for r in selected}
        merged=stage/'merged';(merged/'data').mkdir(parents=True,exist_ok=True);(merged/'labels').mkdir(exist_ok=True)
        records=[];label_records=[]
        for tag, data_records, label_rows, data_dir, label_dir in [
            ('replay',selected,[r for r in capture['records'] if r['sequence'] in names],base/'data',base/'labels'),
            ('fresh',fresh['records'],fresh_capture['records'],stage/'fresh-data',stage/'fresh-labels')]:
            for r in data_records:
                r=dict(r);old=r['path'];r['path']=tag+'-'+old
                (merged/'data'/r['path']).symlink_to((data_dir/old).resolve())
                records.append(r)
            for r in label_rows:
                r=dict(r);old=r['path'];r['path']=tag+'/'+old;r['sequence']=tag+'-'+r['sequence']
                dest=merged/'labels'/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
                dest.symlink_to((label_dir/old).resolve());label_records.append(r)
        (merged/'data/manifest.json').write_text(json.dumps(dict(context=65536,records=records,training_tokens=48*65536),indent=2)+'\n')
        metadata=dict(capture);metadata['records']=label_records
        (merged/'labels/manifest-rank0.json').write_text(json.dumps(metadata,indent=2)+'\n')
        init=['--initial-training-state',previous] if previous else ['--initial-checkpoint',base/'training/epoch-10.pt']
        run('training.train_captured_sidecar','--labels',merged/'labels','--output',stage/'training',
            '--model',args.model,'--epochs','1',*init,gpu=args.gpu)
        previous=stage/'training/resume.pt'
        # Preserve source selections, shard hashes, metrics and checkpoints before
        # deleting this epoch's regenerable fresh activations and merged symlinks.
        shutil.copyfile(merged/'data/manifest.json',stage/'epoch-data-manifest.json')
        shutil.copyfile(merged/'labels/manifest-rank0.json',stage/'epoch-label-manifest.json')
        completion.write_text(json.dumps(dict(epoch=epoch,fresh_streams=24,replay_streams=24,
                                             checkpoint=str(previous)),indent=2)+'\n')
        shutil.rmtree(merged)
        shutil.rmtree(stage/'fresh-labels')
        excludes.append(stage/'fresh-data/manifest.json')
        print(f'Completed progressive epoch {epoch}',flush=True)


if __name__=='__main__':main()
