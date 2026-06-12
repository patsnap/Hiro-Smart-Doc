#!/usr/bin/env python3
"""Download the Hiro-Layout ONNX model(s) from Hugging Face.

The layout weights are not bundled in this repository. They are published at:
    https://huggingface.co/PatSnap/Hiro-Layout

This script fetches the ONNX file(s) into ``LAYOUT_MODEL_DIR`` (default
``./layout_model``) using the expected ``RT-DETR_<id>.onnx`` filename pattern.

Usage:
    uv run python scripts/download_models.py            # downloads model id 25
    uv run python scripts/download_models.py --models 25,9,5

Requires ``huggingface_hub`` (installed via the ``dev`` extra, or ``pip install
huggingface_hub``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HF_REPO_ID = os.getenv("LAYOUT_HF_REPO", "PatSnap/Hiro-Layout")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Hiro-Layout ONNX models.")
    parser.add_argument(
        "--models",
        default=os.getenv("MODEL_LIST", "25"),
        help="Comma-separated layout model ids to download (default: 25).",
    )
    parser.add_argument(
        "--dest",
        default=os.getenv("LAYOUT_MODEL_DIR", "./layout_model"),
        help="Destination directory (default: ./layout_model).",
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO_ID,
        help=f"Hugging Face repo id (default: {HF_REPO_ID}).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n"
            "    uv pip install huggingface_hub\n"
            "or add the dev dependency group.",
            file=sys.stderr,
        )
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_id in model_ids:
        filename = f"RT-DETR_{model_id}.onnx"
        print(f"Downloading {filename} from {args.repo} ...")
        path = hf_hub_download(
            repo_id=args.repo,
            filename=filename,
            local_dir=str(dest),
        )
        print(f"  -> {path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
