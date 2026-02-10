from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    results = []
    for p in input_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        if p.is_symlink():
            continue
        if not p.is_file():
            continue
        results.append(p)
    return results


def load_metadata(metadata_csv: Optional[Path]) -> pd.DataFrame:
    if metadata_csv and metadata_csv.exists():
        return pd.read_csv(metadata_csv)
    return pd.DataFrame(columns=["image_path", "label", "gen", "form"])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(rows: Iterable[dict], output_csv: Path) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(output_csv, index=False)
