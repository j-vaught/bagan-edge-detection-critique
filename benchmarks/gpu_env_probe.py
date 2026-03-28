#!/usr/bin/env python3
"""
Probe CUDA usability inside a SLURM job and save a structured report.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "hardware_probe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"cmd": cmd, "error": repr(exc)}


def main() -> None:
    tag = os.environ.get("PROBE_TAG", "default").strip() or "default"
    payload: dict[str, object] = {
        "tag": tag,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "gpus": os.environ.get("SLURM_GPUS"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": int(torch.cuda.device_count()),
        },
        "nvidia_smi": run_cmd(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]
        ),
    }

    device_reports = []
    for idx in range(torch.cuda.device_count()):
        report = {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "capability": list(torch.cuda.get_device_capability(idx)),
        }
        try:
            start = time.perf_counter()
            a = torch.randn((2048, 2048), device=f"cuda:{idx}")
            b = torch.randn((2048, 2048), device=f"cuda:{idx}")
            c = a @ b
            torch.cuda.synchronize(idx)
            elapsed = time.perf_counter() - start
            report["matmul_ok"] = True
            report["matmul_seconds"] = elapsed
            report["checksum"] = float(c[0, 0].item())
        except Exception as exc:
            report["matmul_ok"] = False
            report["error"] = repr(exc)
        device_reports.append(report)
    payload["devices"] = device_reports

    if not device_reports and not torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
        except Exception as exc:
            payload["cuda_failure"] = repr(exc)

    out_path = RESULTS_DIR / f"gpu_probe_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
