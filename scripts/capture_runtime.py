#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/runtime.json")
    args = ap.parse_args()
    payload = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        payload.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    try:
        payload["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception as exc:
        payload["nvidia_smi_error"] = repr(exc)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
