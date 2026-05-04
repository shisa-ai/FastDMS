"""
shisa-kvquant: KV cache compression with packed HIGGS storage, AQUA-KV prediction, and DMS eviction.

The heavy optional HIGGS/AQUA helpers are loaded lazily so that the core
DMS training scripts can run without optional quantization dependencies.
"""

__version__ = "0.1.0"

__all__ = [
    "HiggsQuantizer",
    "QuantizerBase",
    "QuantizedTensor",
    "PackedHiggsCache",
    "PredictorHiggsCache",
    "TreatPrefixSeparately",
    "SingleChunkQuantizedCacheWithPredictors",
]


def __getattr__(name: str):
    if name in {"HiggsQuantizer", "QuantizerBase", "QuantizedTensor"}:
        from .quantizers import HiggsQuantizer, QuantizerBase, QuantizedTensor

        return {
            "HiggsQuantizer": HiggsQuantizer,
            "QuantizerBase": QuantizerBase,
            "QuantizedTensor": QuantizedTensor,
        }[name]
    if name == "PackedHiggsCache":
        from .packed_cache import PackedHiggsCache

        return PackedHiggsCache
    if name in {"PredictorHiggsCache", "TreatPrefixSeparately"}:
        from .cache import PredictorHiggsCache, TreatPrefixSeparately

        return {
            "PredictorHiggsCache": PredictorHiggsCache,
            "TreatPrefixSeparately": TreatPrefixSeparately,
        }[name]
    if name == "SingleChunkQuantizedCacheWithPredictors":
        from .predictors import SingleChunkQuantizedCacheWithPredictors

        return SingleChunkQuantizedCacheWithPredictors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
