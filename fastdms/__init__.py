from fastdms.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from fastdms.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
