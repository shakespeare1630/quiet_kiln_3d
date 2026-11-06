"""Bounded photometric adjustment of material and vertex colors.

Only nine global hue/brightness gains are fitted from the supplied photo and
witness renders. No photograph texture, benchmark data or fixed output is used.
"""

import colorsys
import io
import json
import math
import re
from dataclasses import replace
from statistics import median

from PIL import Image

from .experimental import EfficientPipeline
from .source import extract


def groups(png):
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image.thumbnail((128, 128), Image.Resampling.BILINEAR)
    width = image.width
    pixels = list(image.get_flattened_data())
    border = pixels[:width] + pixels[-width:] + pixels[::width] + pixels[width - 1 :: width]
    background = [median(p[channel] for p in border) for channel in range(3)]
    result = [[] for _ in range(9)]
    for pixel in pixels:
        if sum(abs(pixel[i] - background[i]) for i in range(3)) < 30:
            continue
        hue, saturation, value = colorsys.rgb_to_hsv(*(component / 255 for component in pixel))
        index = int(hue * 6 + 0.5) % 6 if saturation > 0.2 else 6 if value < 0.2 else 7 if value < 0.7 else 8
        result[index].append(value)
    return result


def pooled(views):
    samples = [groups(view) for view in views]
    return [[value for sample in samples for value in sample[index]] for index in range(9)]


def gains_for(target, current):
    return [
        min(1.35, max(0.75, (median(a) / max(0.01, median(b))) ** 2.2)) if len(a) >= 12 and len(b) >= 12 else 1.0
        for a, b in zip(target, current, strict=True)
    ]


def loss(target, current):
    count = sum(len(values) for values in target)
    if not count:
        return 0.0
    return (
        sum(
            len(a) * (abs(math.log(max(0.01, median(a)) / max(0.01, median(b)))) if b else 1.0)
            for a, b in zip(target, current, strict=True)
            if a
        )
        / count
    )


COLOR_BODY = """
  const gains = __GAINS__;
  function grade(color) {
    const rgb = [color.r, color.g, color.b].map(value => Math.pow(Math.max(0, value), 1/2.2));
    const high = Math.max(...rgb), low = Math.min(...rgb), delta = high-low;
    let bucket;
    if (high > 0 && delta/high > 0.2) {
      let hue = high === rgb[0] ? (rgb[1]-rgb[2])/delta : high === rgb[1] ? (rgb[2]-rgb[0])/delta+2 : (rgb[0]-rgb[1])/delta+4;
      hue = (hue/6+1)%1;
      bucket = Math.floor(hue*6+0.5)%6;
    } else bucket = high < 0.2 ? 6 : high < 0.7 ? 7 : 8;
    color.multiplyScalar(gains[bucket]);
    color.r = Math.min(1, color.r); color.g = Math.min(1, color.g); color.b = Math.min(1, color.b);
  }
  const materials = new Map();
  function graded(material) {
    if (!material || !material.color || material.isShaderMaterial) return material;
    if (materials.has(material)) return materials.get(material);
    const copy = material.clone(); grade(copy.color); materials.set(material, copy); return copy;
  }
  root.traverse(node => {
    if (node.material) node.material = Array.isArray(node.material) ? node.material.map(graded) : graded(node.material);
    if (node.geometry && node.geometry.attributes.color) {
      node.geometry = node.geometry.clone();
      const colors = node.geometry.attributes.color;
      for (let i=0;i<colors.count;i++) {
        const color = new THREE.Color().setRGB(colors.getX(i),colors.getY(i),colors.getZ(i));
        grade(color); colors.setXYZ(i,color.r,color.g,color.b);
      }
      colors.needsUpdate = true;
    }
  });
"""


def color_variant(source, gains):
    # Nest the complete existing module, preserving its helpers and geometry.
    source = extract(source)
    exported = re.search(r"(?m)^[ \t]*export\s+default\s+", source)
    if exported is None:
        raise ValueError("Missing default export")
    prefix, body = source[: exported.start()], source[exported.end() :]
    return (
        "export default function generate(THREE) {\n"
        + prefix
        + "\n const build="
        + body
        + ";\n const root=build(THREE);\n"
        + COLOR_BODY.replace("__GAINS__", json.dumps(gains, allow_nan=False))
        + "\n return root;\n}\n"
    )


class ColorFinisher(EfficientPipeline):
    async def finish(self, parent):
        target = groups(self.photo)
        current = pooled(parent.views)
        gains = gains_for(target, current)
        if max(abs(math.log(gain)) for gain in gains) < 0.08:
            return parent
        baseline = loss(target, current)
        best, best_loss = parent, baseline
        for strength in (0.5, 1.0):
            adjusted = [gain**strength for gain in gains]
            candidate, _ = await self.accept(color_variant(parent.source, adjusted), f"color-{strength}")
            if candidate is None:
                continue
            candidate_loss = loss(target, pooled(candidate.views))
            if candidate_loss < best_loss:
                best, best_loss = candidate, candidate_loss
        if best is parent or baseline - best_loss < max(0.002, baseline * 0.05):
            return parent
        original = self.config
        self.config = replace(original, planner="gpt-6-astra")
        try:
            return best if await self.guard(parent, best) else parent
        finally:
            self.config = original
