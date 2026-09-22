"""Launch one model evaluator without importing GPU libraries in the launcher."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

MODELS = {
    "qwen3-32b": ("qwen.py", "Qwen/Qwen3-VL-32B-Instruct"),
    "qwen3-8b": ("qwen.py", "Qwen/Qwen3-VL-8B-Instruct"),
    "gta1": ("gta1.py", "HelloKKMe/GTA1-7B"),
    "uitars": ("uitars.py", "ByteDance-Seed/UI-TARS-1.5-7B"),
    "uivenus": ("uivenus.py", "inclusionAI/UI-Venus-Ground-7B"),
}
ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI blind-area evaluation; extra flags are passed to the selected evaluator.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("icon", choices=("clock", "gemini", "circle_ok"))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--nproc", type=int, default=1, help="PyTorch worker count; each worker loads a full model")
    args, extra = parser.parse_known_args()
    if args.nproc < 1:
        parser.error("--nproc must be >= 1")
    filename, model_path = MODELS[args.model]
    script = str(ROOT / filename)
    defaults = ["--model_path", model_path,
                "--output_dir", str(ROOT / "results" / args.model)]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.nproc}", script, args.icon, *defaults, *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1")
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
