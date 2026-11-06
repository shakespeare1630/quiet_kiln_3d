import base64
import io
import json

from PIL import Image, ImageOps
from witness_client import RenderOptions, RenderUnavailable

ANGLES = ((30, 0), (330, 0), (0, 15), (0, -30))
ANGLE_NAMES = ("front-left", "front-right", "front-below", "front-above")


def options(pairs=ANGLES, size=512, background="ffffff") -> RenderOptions:
    return RenderOptions(
        thetas=tuple(p[0] for p in pairs),
        phis=tuple(p[1] for p in pairs),
        img_size=size,
        cam_radius=2.0,
        cam_fov_deg=49.1,
        gap=0,
        bg_color=background,
        lighting="neutral",
    )


def encode(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def canonical(image: bytes, size=1024) -> bytes:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGBA")
    background = Image.new("RGBA", im.size, "white")
    im = Image.alpha_composite(background, im).convert("RGB")
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return encode(im)


def data_image(image: bytes) -> dict:
    return {
        "type": "input_image",
        "image_url": "data:image/png;base64," + base64.b64encode(image).decode(),
        "detail": "high",
    }


def text(value: str) -> dict:
    return {"type": "input_text", "text": value}


def panels(png: bytes, size=512) -> tuple[bytes, ...]:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.size != (size * 2, size * 2):
        raise RenderUnavailable("Unexpected four-view grid dimensions")
    return tuple(
        encode(im.crop(((i % 2) * size, (i // 2) * size, (i % 2 + 1) * size, (i // 2 + 1) * size))) for i in range(4)
    )


def color_summary(png: bytes) -> str:
    """Coarse measurements, never a claim of foreground segmentation or material color."""
    im = Image.open(io.BytesIO(png)).convert("RGB")
    im.thumbnail((96, 96), Image.Resampling.BOX)
    pixels = [p for p in im.get_flattened_data() if min(p) < 235 and max(p) - min(p) > 12]
    if not pixels:
        return "No reliable saturated foreground palette; use the reference visually."
    histogram = {}
    for p in pixels:
        key = tuple(min(255, (c // 32) * 32 + 16) for c in p)
        histogram[key] = histogram.get(key, 0) + 1
    palette = sorted(histogram, key=lambda p: (-histogram[p], p))[:5]
    return json.dumps(["#%02x%02x%02x" % p for p in palette])
