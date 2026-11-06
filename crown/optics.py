"""Independent, bounded optical material alternatives for rendered glass.

No lights, backdrop, photo texture or geometry changes. These are material
approximations to test against a supplied image under the real render setup.
"""

import re

from .budget import BudgetStop

VARIANTS = (
    (0.12, 0.85, 0.70),
    (0.35, 0.90, 0.70),
    (0.00, 0.70, 0.50),
    (0.65, 1.00, 0.55),
)

RELATIVE_VARIANTS = ((0.08, 1.20, 0.70), (0.08, 0.70, 0.70), (0.35, 1.00, 0.85), (0.65, 1.00, 0.85))

WRAPPER = """export default function generate(THREE) {
  // crown-optics-v1
  const build = __SOURCE__;
  const root = build(THREE);
  const materials = new Map();
  function adjust(material) {
    if (!material || !material.isMeshPhysicalMaterial || !(material.transmission >= 0.5)) return material;
    if (materials.has(material)) return materials.get(material);
    const copy = material.clone();
    copy.transmission = __TRANSMISSION__;
    copy.opacity = Math.max(copy.opacity, __OPACITY__);
    copy.color.multiplyScalar(__TINT__);
    copy.roughness = Math.max(0.035, Math.min(copy.roughness, 0.16));
    copy.transparent = copy.opacity < 1;
    copy.needsUpdate = true;
    materials.set(material, copy);
    return copy;
  }
  root.traverse(node => {
    if (node.material) node.material = Array.isArray(node.material) ? node.material.map(adjust) : adjust(node.material);
  });
  return root;
}
"""


def optical_variant(source, transmission, opacity, tint):
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in (transmission, opacity, tint)):
        raise ValueError("Optical parameters must be finite values in [0,1]")
    return _variant(source, transmission, opacity, tint, WRAPPER)


def relative_optical_variant(source, transmission, opacity_scale, tint):
    if (
        type(opacity_scale) not in (int, float)
        or not 0 < opacity_scale <= 2
        or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in (transmission, tint))
    ):
        raise ValueError("Invalid bounded relative optical parameters")
    wrapper = WRAPPER.replace("crown-optics-v1", "crown-optics-v2").replace(
        "Math.max(copy.opacity, __OPACITY__)", "Math.min(0.98, copy.opacity * __OPACITY__)"
    )
    return _variant(source, transmission, opacity_scale, tint, wrapper)


def _variant(source, transmission, opacity, tint, wrapper):
    match = re.search(r"(?m)^[ \t]*export\s+default\s+", source)
    if match is None:
        raise ValueError("Missing default export")
    prefix, body = source[: match.start()], source[match.end() :]
    template = (
        wrapper.replace("__SOURCE__", body)
        if not prefix
        else wrapper.replace("  const build = __SOURCE__;", prefix + "\n  const build = " + body + ";")
    )
    return (
        template.replace("__TRANSMISSION__", str(transmission))
        .replace("__OPACITY__", str(opacity))
        .replace("__TINT__", str(tint))
    )


async def finish_optics(pipeline, parent, *, variants=VARIANTS, factory=optical_variant):
    """Rank complete small pools; preserve the original unless its guard passes."""
    selected = parent
    for start in range(0, len(variants), 3):
        candidates = [selected]
        for index, params in enumerate(variants[start : start + 3], start):
            source = factory(parent.source, *params)
            child, _ = await pipeline.accept(source, f"optics-{index}")
            if child is not None and all(child.front != candidate.front for candidate in candidates):
                candidates.append(child)
        try:
            selected = await pipeline.select(candidates)
        except BudgetStop:
            return parent
    if selected is not parent and await pipeline.guard(parent, selected):
        return selected
    return parent
