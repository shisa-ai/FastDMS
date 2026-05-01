"""P0.2 retained comparison artifact generator.

Diffs two bench_llama_wikitext2.py artifacts (one FastDMS row, one mini-sglang
row) and prints prefill/decode/memory ratios. Names the attention backend on
each side so the comparison cannot be silently contaminated by backend drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    if rows:
        return rows[0]
    return payload


def _attention_backend(payload: dict[str, Any]) -> str | None:
    backend = (
        payload.get("attention_backend")
        or payload.get("attention_kernel")
        or payload.get("env", {}).get("attention_backend")
        or payload.get("runtime", {}).get("flash_attn")
    )
    if backend is None:
        runtime = payload.get("runtime", {}) or {}
        for key in ("flash_attn_3", "flashinfer", "flash_attn"):
            value = runtime.get(key) if isinstance(runtime, dict) else None
            if value:
                return f"{key}={value}"
    return backend


def _engine(payload: dict[str, Any]) -> str:
    return payload.get("engine") or payload.get("rows", [{}])[0].get("engine") or "unknown"


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two bench artifacts.")
    parser.add_argument("--reference", type=Path, required=True, help="Anchor side (e.g. mini-sglang).")
    parser.add_argument("--candidate", type=Path, required=True, help="Other side (e.g. FastDMS).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ref_payload = _read(args.reference)
    cand_payload = _read(args.candidate)
    ref_row = _row(ref_payload)
    cand_row = _row(cand_payload)

    summary = {
        "reference": {
            "path": str(args.reference),
            "engine": _engine(ref_payload),
            "name": ref_row.get("name"),
            "concurrency": ref_payload.get("concurrency"),
            "ctx_len": ref_payload.get("ctx_len"),
            "gen_len": ref_payload.get("gen_len"),
            "attention_backend": _attention_backend(ref_payload),
            "kv_cache_dtype": ref_row.get("kv_cache_dtype"),
            "compact_kv_enabled": ref_row.get("compact_kv_enabled"),
            "prefill_tok_s": ref_row.get("prefill_tok_s"),
            "decode_tok_s": ref_row.get("decode_tok_s"),
            "kv_total_gib": ref_row.get("kv_total_gib"),
            "effective_total_gib": ref_row.get("effective_total_gib"),
            "allocator_visible_total_gib": ref_row.get("allocator_visible_total_gib"),
            "torch_reserved_gib": ref_row.get("torch_reserved_gib"),
        },
        "candidate": {
            "path": str(args.candidate),
            "engine": _engine(cand_payload),
            "name": cand_row.get("name"),
            "concurrency": cand_payload.get("concurrency"),
            "ctx_len": cand_payload.get("ctx_len"),
            "gen_len": cand_payload.get("gen_len"),
            "attention_backend": _attention_backend(cand_payload),
            "kv_cache_dtype": cand_row.get("kv_cache_dtype"),
            "compact_kv_enabled": cand_row.get("compact_kv_enabled"),
            "prefill_tok_s": cand_row.get("prefill_tok_s"),
            "decode_tok_s": cand_row.get("decode_tok_s"),
            "kv_total_gib": cand_row.get("kv_total_gib"),
            "effective_total_gib": cand_row.get("effective_total_gib"),
            "allocator_visible_total_gib": cand_row.get("allocator_visible_total_gib"),
            "torch_reserved_gib": cand_row.get("torch_reserved_gib"),
        },
        "ratios_candidate_over_reference": {
            "prefill_tok_s": _safe_ratio(cand_row.get("prefill_tok_s"), ref_row.get("prefill_tok_s")),
            "decode_tok_s": _safe_ratio(cand_row.get("decode_tok_s"), ref_row.get("decode_tok_s")),
            "kv_total_gib": _safe_ratio(cand_row.get("kv_total_gib"), ref_row.get("kv_total_gib")),
            "effective_total_gib": _safe_ratio(cand_row.get("effective_total_gib"), ref_row.get("effective_total_gib")),
            "allocator_visible_total_gib": _safe_ratio(
                cand_row.get("allocator_visible_total_gib"),
                ref_row.get("allocator_visible_total_gib"),
            ),
            "torch_reserved_gib": _safe_ratio(cand_row.get("torch_reserved_gib"), ref_row.get("torch_reserved_gib")),
        },
        "comparison_caveats": [
            "Attention backends must be named on both sides; treat the ratio as contaminated otherwise.",
            "Memory ratio is meaningful only when token-pool size matches (token_pool field).",
        ],
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
