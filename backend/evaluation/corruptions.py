"""Repeatable image corruptions for the robustness tests. All take and return RGB PIL images."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def brightness(image: Image.Image, factor: float, rng=None) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def low_light(image: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """Darkens with a gamma curve and adds sensor-like noise, which plain dimming lacks."""
    values = np.asarray(image, dtype=np.float32) / 255
    values = (values**2.2) * factor
    values += rng.normal(0, 0.01 + 0.02 * (1 - factor), values.shape)
    return Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8))


def blur(image: Image.Image, radius: float, rng=None) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    values = np.asarray(image, dtype=np.float32) + rng.normal(0, sigma, (image.height, image.width, 3))
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def low_resolution(image: Image.Image, longest_side: int, rng=None) -> Image.Image:
    """Shrinks and scales back up, so the detail is lost but the input size is unchanged."""
    scale = longest_side / max(image.size)
    if scale >= 1:
        return image
    small = image.resize((max(int(image.width * scale), 1), max(int(image.height * scale), 1)), Image.BILINEAR)
    return small.resize(image.size, Image.BILINEAR)


def occlusion(image: Image.Image, area_fraction: float, rng: np.random.Generator) -> Image.Image:
    """A grey rectangle covering the given share of the image, at a random position."""
    side = np.sqrt(area_fraction)
    width, height = int(image.width * side), int(image.height * side)
    x = int(rng.integers(0, max(image.width - width, 1)))
    y = int(rng.integers(0, max(image.height - height, 1)))
    output = image.copy()
    ImageDraw.Draw(output).rectangle((x, y, x + width, y + height), fill=(128, 128, 128))
    return output


CORRUPTIONS: dict[str, tuple[Callable, tuple]] = {
    "brightness": (brightness, (1.5, 2.0, 2.5)),
    "low_light": (low_light, (0.6, 0.35, 0.15)),
    "blur": (blur, (1.0, 2.5, 5.0)),
    "noise": (noise, (15.0, 35.0, 60.0)),
    "low_resolution": (low_resolution, (128, 64, 32)),
    "occlusion": (occlusion, (0.1, 0.25, 0.4)),
}


def apply(name: str, severity: int, image: Image.Image, seed: int) -> Image.Image:
    """Severity is 0, 1 or 2. The same seed always produces the same result."""
    function, levels = CORRUPTIONS[name]
    return function(image, levels[severity], np.random.default_rng(seed))
