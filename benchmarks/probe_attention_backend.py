"""P0.4 retained backend-version probe.

Records the attention-backend kernels each side of a comparison actually used.
Run on each environment and keep the JSON next
to the comparison artifact so backend drift is mechanically named.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_info(name: str) -> dict | None:
    try:
        module = __import__(name)
    except Exception as exc:
        return {"name": name, "import_error": repr(exc)}
    info = {
        "name": name,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
        "metadata_version": _version(name.replace("_", "-")),
    }
    package_root = info["file"]
    if package_root:
        info["package_dir"] = str(Path(package_root).resolve().parent)
    return info


def _git_info(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=p, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        remotes = subprocess.run(
            ["git", "remote", "-v"], cwd=p, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        remotes = None
    return {"path": str(p), "commit": commit, "remotes": remotes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe attention-backend versions and origins.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch  # local import so import failure is surfaced

    info = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "modules": {},
    }
    for name in ("flash_attn", "flash_attn_2", "flash_attn_3", "flashinfer", "vllm", "sglang", "triton", "transformers"):
        result = _module_info(name)
        if result is not None:
            info["modules"][name] = result
            info["modules"][name]["git"] = _git_info(result.get("package_dir"))

    text = json.dumps(info, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
