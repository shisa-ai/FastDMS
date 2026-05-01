from collections import deque

from fastdms.config import Config
from fastdms.engine.sequence import Sequence, SequenceStatus
from fastdms.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.compact_kv_enabled = config.compact_kv_enabled
        # J1: streaming pack mode = compact + DMS, no dense KV blocks. Block
        # manager has 0 blocks; admission goes through CompactKVManager only.
        self.streaming_pack_mode = (
            config.compact_kv_enabled and config.compact_kv_retention_mode == "dms"
        )
        self.block_manager = BlockManager(max(1, config.num_kvcache_blocks), config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.freed_seq_ids: list[int] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if self.streaming_pack_mode:
                # J1: no dense KV admission; CompactKVManager.allocate is the
                # gate, called from prepare_prefill. Just respect token budget.
                if remaining == 0:
                    break
                if remaining < num_tokens and scheduled_seqs:
                    break
                seq.num_scheduled_tokens = min(num_tokens, remaining)
                if seq.num_scheduled_tokens == num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += seq.num_scheduled_tokens
                continue
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
                break
            if remaining < num_tokens and scheduled_seqs:    # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += seq.num_scheduled_tokens
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if self.compact_kv_enabled:
                # B1.2: compact decode does not need new dense blocks; the dense
                # arena was released after pack_compact_prefill. Compact-arena
                # admission is enforced by CompactKVManager.allocate at prefill,
                # not here.
                seq.num_scheduled_tokens = 1
                scheduled_seqs.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.freed_seq_ids.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                self.freed_seq_ids.append(seq.seq_id)
                self.running.remove(seq)
        # B1.2: after a compact prefill, release the per-seq dense blocks so the
        # block_manager free pool is replenished for the next prefill batch.
        # Compact decode does not need dense blocks; CompactKVManager now owns
        # the KV state. In J1 streaming-pack mode there are no dense blocks
        # to deallocate.
        if is_prefill and self.compact_kv_enabled and not self.streaming_pack_mode:
            for seq in seqs:
                if seq.block_table and seq.status == SequenceStatus.RUNNING:
                    self.block_manager.deallocate(seq)
                    seq.block_table = []
                    seq.num_cached_tokens = seq.num_tokens

    def consume_freed_seq_ids(self) -> list[int]:
        freed_seq_ids = self.freed_seq_ids
        self.freed_seq_ids = []
        return freed_seq_ids
