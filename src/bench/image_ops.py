from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class VariantConfig:
    image_size: int = 224
    background: str = "white"
    foreground: str = "black"
    alpha_threshold: int = 10
    contour_width: int = 2
    occlusion_ratio: float = 0.2
    partial_reveal_ratio: float = 0.3
    noise_sigma: float = 8.0


def _resize_keep_aspect(img: Image.Image, size: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _alpha_mask(img: Image.Image, threshold: int) -> np.ndarray:
    rgba = np.array(img)
    alpha = rgba[:, :, 3]
    return (alpha > threshold).astype(np.uint8)


def _to_silhouette(mask: np.ndarray, cfg: VariantConfig) -> Image.Image:
    h, w = mask.shape
    bg = (255, 255, 255) if cfg.background == "white" else (0, 0, 0)
    fg = (0, 0, 0) if cfg.foreground == "black" else (255, 255, 255)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = bg
    out[mask == 1] = fg
    return Image.fromarray(out, mode="RGB")


def make_silhouette(img: Image.Image, cfg: VariantConfig) -> Image.Image:
    img = _resize_keep_aspect(img, cfg.image_size)
    mask = _alpha_mask(img, cfg.alpha_threshold)
    return _to_silhouette(mask, cfg)


def make_edge(img: Image.Image, cfg: VariantConfig) -> Image.Image:
    img = _resize_keep_aspect(img, cfg.image_size)
    mask = _alpha_mask(img, cfg.alpha_threshold)
    edge = np.zeros_like(mask)
    edge[1:, :] |= mask[1:, :] ^ mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] ^ mask[:, :-1]
    for _ in range(max(1, cfg.contour_width - 1)):
        edge = _dilate(edge)
    return _to_silhouette(edge, cfg)


def make_occlusion(img: Image.Image, cfg: VariantConfig) -> Image.Image:
    sil = make_silhouette(img, cfg)
    arr = np.array(sil)
    h, w, _ = arr.shape
    occ_h = int(h * cfg.occlusion_ratio)
    occ_w = int(w * cfg.occlusion_ratio)
    y0 = np.random.randint(0, max(1, h - occ_h))
    x0 = np.random.randint(0, max(1, w - occ_w))
    color = (255, 255, 255) if cfg.background == "white" else (0, 0, 0)
    arr[y0 : y0 + occ_h, x0 : x0 + occ_w] = color
    return Image.fromarray(arr, mode="RGB")


def make_partial(img: Image.Image, cfg: VariantConfig) -> Image.Image:
    sil = make_silhouette(img, cfg)
    arr = np.array(sil)
    h, w, _ = arr.shape
    total = h * w
    keep = int(total * cfg.partial_reveal_ratio)
    idx = np.random.choice(total, keep, replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[idx] = True
    mask = mask.reshape(h, w)
    bg = (255, 255, 255) if cfg.background == "white" else (0, 0, 0)
    arr[~mask] = bg
    return Image.fromarray(arr, mode="RGB")


def make_noise(img: Image.Image, cfg: VariantConfig) -> Image.Image:
    sil = make_silhouette(img, cfg)
    arr = np.array(sil).astype(np.float32)
    noise = np.random.normal(0, cfg.noise_sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x] == 1:
                out[y - 1 : y + 2, x - 1 : x + 2] = 1
    return out


VARIANT_FUNCS = {
    "silhouette": make_silhouette,
    "edge": make_edge,
    "occlusion": make_occlusion,
    "partial": make_partial,
    "noise": make_noise,
}
