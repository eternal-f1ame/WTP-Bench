#!/usr/bin/env python
"""OpenRouter multimodal inference (GPT/Claude/Gemini/Grok)."""
from __future__ import annotations

import argparse
import base64
import io
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM inference via OpenRouter API.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Identify this Pokemon. Respond with only the Pokemon's name, nothing else.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--image-col", type=str, default="image_path")
    parser.add_argument("--api-base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--referer", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--reasoning-effort", type=str, default="",
                        help="Reasoning effort level (low/medium/high). Empty = omit from request.")
    return parser.parse_args()


def resolve_image_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        base_dir / raw_path,
        Path(raw_path),
        base_dir / raw_path.removeprefix("data/") if raw_path.startswith("data/") else None,
    ]

    for c in candidates:
        if c and c.exists():
            return c

    raise FileNotFoundError(f"Could not find image: {raw_path}")


RESIZE_SIZE = (200, 200)


def image_to_base64_png(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(RESIZE_SIZE)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return str(content).strip()


def call_openrouter(
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    prompt: str,
    image_b64: str,
    max_tokens: int,
    temperature: float,
    referer: str,
    title: str,
    reasoning_effort: str = "",
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        print(f"[openrouter] HTTP {resp.status_code}: {resp.text}", flush=True)
    resp.raise_for_status()
    data = resp.json()
    text = _extract_text(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    # Enforce max_tokens: truncate to at most max_tokens words as a safety net
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])
    return text


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in env var: {args.api_key_env}")

    df = pd.read_csv(args.input_csv)
    if args.image_col not in df.columns:
        raise ValueError(f"Input CSV must include {args.image_col} column.")

    if args.limit > 0:
        df = df.head(args.limit)

    # Resume: load existing predictions if output CSV already exists
    done: dict[str, dict] = {}
    if args.output_csv.exists():
        prev = pd.read_csv(args.output_csv)
        for _, r in prev.iterrows():
            done[str(r["image_path"])] = r.to_dict()
        print(f"[openrouter] Resuming: {len(done)} images already done, {len(df) - len(done)} remaining")

    rows = list(done.values())
    for _, row in df.iterrows():
        raw_path = str(row[args.image_col]).lstrip("/")
        if raw_path in done:
            continue

        image_path = resolve_image_path(args.base_dir, raw_path)
        image_b64 = image_to_base64_png(image_path)

        pred = call_openrouter(
            api_base=args.api_base,
            api_key=api_key,
            model_id=args.model_id,
            prompt=args.prompt,
            image_b64=image_b64,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            referer=args.referer,
            title=args.title,
            reasoning_effort=args.reasoning_effort,
        )

        print(f"[openrouter] {raw_path} -> {pred}")

        rows.append(
            {
                "image_path": raw_path,
                "label": row.get("label", ""),
                "prediction": pred,
            }
        )

        # Write after each prediction so progress is never lost
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
