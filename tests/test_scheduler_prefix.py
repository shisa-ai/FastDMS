from types import SimpleNamespace
from fastdms.engine.scheduler import Scheduler
from fastdms.engine.sequence import Sequence
from fastdms.sampling_params import SamplingParams


def test_cached_prefix_schedules_only_remaining_tokens():
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        compact_kv_enabled=False, compact_kv_retention_mode='all',
        num_kvcache_blocks=8, kvcache_block_size=256)
    scheduler = Scheduler(config)
    first = Sequence(list(range(512)), SamplingParams(max_tokens=1))
    scheduler.block_manager.allocate(first)
    scheduler.block_manager.deallocate(first)
    second = Sequence(list(range(512)), SamplingParams(max_tokens=1))
    scheduler.add(second)
    seqs, prefill = scheduler.schedule()
    assert prefill
    assert second.num_scheduled_tokens == 1
    assert min(second.num_cached_tokens, len(second)-1) + second.num_scheduled_tokens == len(second)
