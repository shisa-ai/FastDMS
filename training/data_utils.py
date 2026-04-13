"""
Calibration data loaders for predictor training.

Supports wikitext2, c4, ptb, pajama, and custom pre-tokenized datasets.
Based on https://github.com/IST-DASLab/gptq and https://github.com/Vahe1994/AQLM.
"""
import os
import random
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import trange
from transformers import AutoTokenizer


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            trainloader.append(trainenc.input_ids[:, i:j])
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")


def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        trainloader.append(trainenc.input_ids[:, i : i + seqlen])
    return trainloader


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            trainloader.append(trainenc.input_ids[:, i : i + seqlen])
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        return tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    c4_revision = "607bd4c8450a42878aa9ddc051a65a055450ef87"
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4", "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train", revision=c4_revision,
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            trainloader.append(trainenc.input_ids[:, i : i + seqlen])
        return trainloader
    else:
        valdata = load_dataset(
            "allenai/c4", "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation", revision=c4_revision,
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                valenc.append(tmp.input_ids[:, i : i + seqlen])
        return torch.hstack(valenc)


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    """Load calibration or evaluation data for a Transformers model."""
    set_seed(seed)

    if name.lower() == "none":
        print("Not loading any dataset.")
        return None
    elif os.path.isfile(name):
        data = torch.load(name, weights_only=True)[:nsamples]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )
        loaders = {
            "wikitext2": get_wikitext2,
            "pajama": get_red_pajama,
            "ptb": get_ptb,
            "c4": get_c4,
        }
        loader_fn = loaders.get(name.lower())
        if loader_fn is None:
            raise ValueError(f"Unknown dataset: {name}. Use one of {list(loaders.keys())} or a file path.")
        data = loader_fn(nsamples, seqlen, tokenizer, eval_mode=eval_mode)

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data
