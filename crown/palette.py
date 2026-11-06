"""Measured colour palettes: the photo's object colours and a candidate's judge-view colours, as exact hex values.

Vision models judge colour by eye and drift (wrong blue, washed-out red); exact measured values give the code
writer a numeric target and a numeric check. On the F1 benchmark the per-duel difference in colour-histogram
similarity to the photo correlated with the judge's verdict (+0.32 over 82 decided duels).

Everything here is integer arithmetic plus Pillow's integer resampling/median-cut quantizer, so the prompt text
is identical on every machine (replay compares request bodies byte for byte).
"""

import io

import numpy as np
from PIL import Image, ImageFilter

PHOTO_SIZE = 192
RENDER_SIZE = 256  # per judge view
BINS = 6
MAX_COLOURS = 6
MIN_SHARE = 4  # percent


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int32)


def _clean(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8), "L")
    image = image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return np.asarray(image) > 127


def photo_pixels(photo: bytes) -> np.ndarray:
    """Object pixels of a product photo on a plain light background (border colour removed, soft shadows dropped)."""
    image = Image.open(io.BytesIO(photo)).convert("RGB").resize((PHOTO_SIZE, PHOTO_SIZE), Image.Resampling.BOX)
    rgb = _array(image)
    ring = np.concatenate(
        [rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3), rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)]
    )
    background = np.sort(ring, axis=0)[len(ring) // 2]
    distance2 = ((rgb - background) ** 2).sum(axis=2)
    luma = (rgb * np.array([299, 587, 114])).sum(axis=2) // 1000
    background_luma = int((background * np.array([299, 587, 114])).sum()) // 1000
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    shadow = (chroma < 16) & (luma < background_luma) & (background_luma - luma < 45) & (distance2 < 70 * 70)
    mask = _clean((distance2 > 30 * 30) & ~shadow)
    if mask.sum() < rgb.shape[0] * rgb.shape[1] // 50:
        quarter = PHOTO_SIZE // 4
        return rgb[quarter:-quarter, quarter:-quarter].reshape(-1, 3)
    return rgb[mask]


def render_pixels(front_png: bytes) -> np.ndarray:
    """Object pixels of the four judge views (2x2 sheet on white)."""
    image = Image.open(io.BytesIO(front_png)).convert("RGB")
    image = image.resize((2 * RENDER_SIZE, 2 * RENDER_SIZE), Image.Resampling.BOX)
    rgb = _array(image)
    mask = _clean(((255 - rgb) ** 2).sum(axis=2) > 20 * 20)
    return rgb[mask] if mask.any() else rgb.reshape(-1, 3)


def palette(pixels: np.ndarray) -> list[tuple[str, int]]:
    """Up to six dominant colours as (hex, integer percent of pixels), largest first."""
    count = len(pixels)
    if count == 0:
        return []
    strip = Image.fromarray(pixels.astype(np.uint8).reshape(1, count, 3), "RGB")
    quantized = strip.quantize(colors=12, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    colours = quantized.getpalette()[: 3 * 12]
    tally = sorted(quantized.getcolors(), key=lambda item: (-item[0], item[1]))
    merged: list[list[int]] = []  # [weight, r, g, b] as weighted sums
    for weight, index in tally:
        rgb = colours[3 * index : 3 * index + 3]
        for entry in merged:
            centre = [entry[1 + c] // entry[0] for c in range(3)]
            if max(abs(centre[c] - rgb[c]) for c in range(3)) < 24:
                entry[0] += weight
                for c in range(3):
                    entry[1 + c] += rgb[c] * weight
                break
        else:
            merged.append([weight, *(value * weight for value in rgb)])
    merged.sort(key=lambda entry: -entry[0])
    out = []
    for entry in merged[:MAX_COLOURS]:
        share = 100 * entry[0] // count
        if share < MIN_SHARE:
            continue
        out.append(("#" + "".join(f"{entry[1 + c] // entry[0]:02x}" for c in range(3)), share))
    return out


def histogram(pixels: np.ndarray) -> np.ndarray:
    quant = pixels.astype(np.int64) * BINS // 256
    return np.bincount((quant[:, 0] * BINS + quant[:, 1]) * BINS + quant[:, 2], minlength=BINS**3)


def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Colour-histogram intersection of two pixel sets, integer percent (100 = identical colour distribution)."""
    ha, hb = histogram(a), histogram(b)
    na, nb = int(ha.sum()), int(hb.sum())
    if not na or not nb:
        return 0
    return int(100 * int(np.minimum(ha * nb, hb * na).sum()) // (na * nb))


def describe(colours: list[tuple[str, int]]) -> str:
    return ", ".join(f"{hex_} {share}%" for hex_, share in colours)
