"""Geometry feedback at a consistent close-up scale, preserving complete outlines."""

import io

from PIL import Image, ImageChops

from . import media
from .adaptive import AdaptiveDetailPipeline
from .detail import RANK
from .optics import RELATIVE_VARIANTS, finish_optics, relative_optical_variant

OPTICS_RANK = """Compare material likeness against the reference photograph using all
four white views and the gray view. Geometry is mostly shared; do not reward an
added dark tint just for making an outline easier to see. Compare whether the
reference is clear, translucent, frosted or opaque: background visibility through
the object, edge reflections, highlights, color neutrality and thickness cues.
Opaque gray walls are a defect when the reference is clear glass. Nearly invisible
walls are also a defect when reference contours are visible. Preserve differences
between thin walls, inner surfaces, a heavy base and rims. Judge overall likeness,
not visibility alone. Return JSON only with `penalties` mapping EVERY stable
candidate ID to a finite 0-10 penalty, and a short `issues` string. 0 faithful,
3 small discrepancies, 5 substantial errors, 7 wrong related object, 10 unrelated.
Ignore framing, background and incidental lighting; IDs stay attached in both orders."""


def tight_views(white, gray):
    """Use one common crop for all views; gray recovers faint white-background edges."""
    images = [Image.open(io.BytesIO(png)).convert("RGB") for png in (*white, gray)]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("Close-up views have different dimensions")
    boxes = []
    for index, image in enumerate(images):
        color = (255, 255, 255) if index < len(white) else (128, 128, 128)
        channels = ImageChops.difference(image, Image.new("RGB", image.size, color)).split()
        difference = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        box = difference.point(lambda value: 255 if value > 5 else 0).getbbox()
        if box:
            boxes.append(box)
    if not boxes:
        return tuple(white), gray
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    extent = max(right - left, bottom - top)
    side = min(min(width, height), extent + 2 * max(12, round(extent * 0.06)))
    x = max(0, min(width - side, (left + right - side) // 2))
    y = max(0, min(height - side, (top + bottom - side) // 2))
    crop = (x, y, x + side, y + side)
    encoded = [media.encode(image.crop(crop).resize((1024, 1024), Image.Resampling.LANCZOS)) for image in images]
    return tuple(encoded[:-1]), encoded[-1]


class CloseupPipeline(AdaptiveDetailPipeline):
    finishing_reserve_credits = 500_000

    def ranking_instructions(self):
        return OPTICS_RANK if getattr(self, "optical_pass", False) else RANK

    async def detail_views(self, candidate):
        white, gray = await super().detail_views(candidate)
        return tight_views(white, gray)

    async def finish(self, parent):
        self.optical_pass = True
        try:
            return await finish_optics(self, parent, variants=RELATIVE_VARIANTS, factory=relative_optical_variant)
        finally:
            self.optical_pass = False
