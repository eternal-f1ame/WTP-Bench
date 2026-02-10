#!/usr/bin/env python
"""Quick test runner for local and API-backed models."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.scripts.vlm_infer_openrouter import (
    call_openrouter,
    image_to_base64_png,
    resolve_image_path,
)
from src.scripts.vlm_infer_anthropic import call_anthropic_batch

DEFAULT_PROMPT = "Identify this Pokemon. Respond with only the Pokemon's name, nothing else."


def find_backend(model_id: str, configs_dir: Path) -> str:
    for path in configs_dir.rglob("*.json"):
        try:
            cfg = json.loads(path.read_text())
        except Exception:
            continue
        if cfg.get("model_id") == model_id:
            return str(cfg.get("backend", "hf"))
    return "hf"


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a single model on a few images.")
    parser.add_argument("model_id", help="Model ID to test")
    parser.add_argument("num_images", type=int, nargs="?", default=3)
    args = parser.parse_args()

    model_id = args.model_id
    num = args.num_images

    configs_dir = Path("configs")
    backend = find_backend(model_id, configs_dir)

    df = pd.read_csv("data/metadata.csv").head(num)

    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Missing OPENROUTER_API_KEY in environment.")

        rows = []
        for _, row in df.iterrows():
            raw_path = str(row["image_path"]).lstrip("/")
            image_path = resolve_image_path(Path("."), raw_path)
            image_b64 = image_to_base64_png(image_path)
            pred = call_openrouter(
                api_base="https://openrouter.ai/api/v1",
                api_key=api_key,
                model_id=model_id,
                prompt=DEFAULT_PROMPT,
                image_b64=image_b64,
                max_tokens=16,
                temperature=0.0,
                referer="",
                title="",
            )
            rows.append({"image_path": raw_path, "prediction": pred})
        print(pd.DataFrame(rows).to_string(index=False))
    elif backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Missing ANTHROPIC_API_KEY in environment.")

        rows = []
        for _, row in df.iterrows():
            image_path = str(Path(".") / row["image_path"])
            preds = call_anthropic_batch(
                model_id=model_id,
                image_paths=[image_path],
                prompt=DEFAULT_PROMPT,
                max_tokens=16,
                temperature=0.0,
            )
            pred = preds[0] if isinstance(preds, list) and preds else preds
            rows.append({"image_path": row["image_path"], "prediction": pred})
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        run_cmd(
            [
                sys.executable,
                "src/scripts/vlm_infer_template.py",
                "--input-csv",
                "data/metadata.csv",
                "--output-csv",
                "/dev/null",
                "--model-id",
                model_id,
                "--base-dir",
                ".",
                "--image-col",
                "image_path",
                "--device",
                "auto",
                "--dtype",
                "bfloat16",
                "--max-new-tokens",
                "16",
                "--batch-size",
                "1",
                "--test",
                str(num),
            ]
        )
        print(f"Test completed for {model_id} ({backend}).")


if __name__ == "__main__":
    main()
