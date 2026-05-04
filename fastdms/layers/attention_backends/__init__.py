"""H1.2 attention-backend abstraction.

Today FastDMS has one attention backend: ``flash_attn`` 2.8.x.

The plan calls for the ability to swap in FlashInfer (FP8 paged decode) and
FlashAttention 3 once available on the local environment. The flashinfer
package has been verified importable in the kvcache env (0.6.7.post3) but a
full port requires changing the dense KV storage layout from
``(2, layers, blocks, block_size, num_kv_heads, head_dim)`` to FlashInfer's
``(num_pages, 2, page_size, num_kv_heads, head_dim)`` plus per-layer plan()
calls. That is a follow-up plan; documented here so the abstraction point
is in place when we pick it up.

FlashAttention 3 is not present on this env (not on PyPI for cu130), so the
abstraction also leaves a placeholder branch.

The current production backend is ``flash_attn`` 2.8.4.
"""
from __future__ import annotations

import os


_PRODUCTION_BACKEND = os.environ.get("FASTDMS_ATTENTION_BACKEND", "flash_attn")
_AVAILABLE_BACKENDS = ("flash_attn",)  # flashinfer + fa3 deferred per H1 plan


def get_backend_name() -> str:
    if _PRODUCTION_BACKEND not in _AVAILABLE_BACKENDS:
        raise RuntimeError(
            f"FASTDMS_ATTENTION_BACKEND={_PRODUCTION_BACKEND!r} is not implemented. "
            f"Available: {_AVAILABLE_BACKENDS}. FlashInfer + FA3 ports are deferred per "
            "PLAN15 H1 (storage layout change required)."
        )
    return _PRODUCTION_BACKEND


def list_available_backends() -> tuple[str, ...]:
    return _AVAILABLE_BACKENDS
