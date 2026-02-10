#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from bench.label_utils import canonicalize_label, extract_pokemon_name, extract_pokemon_names_topk

import warnings
warnings.filterwarnings("ignore")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV.")
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--pred-col", type=str, default="prediction")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--raw", action="store_true", help="Treat predictions as raw names (no extraction)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.pred_csv)

    if args.label_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError("CSV must include label and prediction columns.")

    labels = df[args.label_col].fillna("").apply(canonicalize_label)
    preds = df[args.pred_col].fillna("")

    correct = 0
    total = len(df)

    for label, pred in zip(labels, preds):
        pred_str = str(pred)

        if args.raw:
            # Raw mode: expect pipe-delimited names or single name
            if "|" in pred_str:
                extracted = [p.strip() for p in pred_str.split("|") if p.strip()][:args.topk]
            else:
                extracted = [pred_str.strip()] if pred_str.strip() else []
        else:
            # Extraction mode: parse VLM response to extract Pokemon names
            if args.topk > 1:
                extracted = extract_pokemon_names_topk(pred_str, args.topk)
            else:
                name = extract_pokemon_name(pred_str)
                extracted = [name] if name else []

        # Canonicalize extracted names for comparison
        extracted_canonical = [canonicalize_label(n) for n in extracted]

        if label in extracted_canonical:
            correct += 1

    acc = correct / total if total else 0.0
    print(f"Top-{args.topk} accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
