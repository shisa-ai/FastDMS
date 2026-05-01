from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class PhaseProfiler:
    """CUDA-event timer for GPU phases; wallclock for CPU-only phases.

    Cheap when disabled. Pairs of CUDA events are buffered and converted to
    milliseconds only when summarize() is called (a single torch.cuda.synchronize
    flushes the queue). Wallclock phases accumulate directly.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.totals_ms: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        self.totals_ms.clear()
        self.counts.clear()
        self._pending.clear()

    @contextmanager
    def cuda_phase(self, name: str):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))
            self.counts[name] = self.counts.get(name, 0) + 1

    @contextmanager
    def wall_phase(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.totals_ms[name] = self.totals_ms.get(name, 0.0) + elapsed_ms
            self.counts[name] = self.counts.get(name, 0) + 1

    def summarize(self) -> dict:
        if self._pending:
            torch.cuda.synchronize()
            for name, start, end in self._pending:
                ms = start.elapsed_time(end)
                self.totals_ms[name] = self.totals_ms.get(name, 0.0) + ms
            self._pending.clear()
        return {
            "totals_ms": dict(self.totals_ms),
            "counts": dict(self.counts),
        }


_PROFILER = PhaseProfiler()


def get_profiler() -> PhaseProfiler:
    return _PROFILER
