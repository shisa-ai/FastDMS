import torch
from benchmarks.check_nanbeige_kv_quality import cases


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def test_quality_cases_are_fixed_and_retrieval_is_not_truncated():
    first = list(cases(ByteTokenizer()))
    assert first == list(cases(ByteTokenizer()))
    assert len(first) == 7
    for name, prompt, target in first:
        assert len(target) > 1
        if name.startswith('retrieval'):
            assert len(prompt) == int(name.split('-')[1])
            assert bytes(prompt).count(b'The access code for project Cedar is 739281.') == 1
            assert bytes(target) == b'739281'
