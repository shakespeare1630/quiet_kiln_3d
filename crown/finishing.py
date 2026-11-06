"""Guarded material alternatives over existing geometry, for development trials."""

import re

from .budget import BudgetStop
from .experimental import EfficientPipeline

MATERIAL_WRAPPER = """export default function generate(THREE) {
  const build = __SOURCE__;
  const root = build(THREE);
  const copies = new Map();
  function adjusted(material) {
    if (!material || (!material.isMeshStandardMaterial && !material.isMeshPhongMaterial)) return material;
    if (copies.has(material)) return copies.get(material);
    const copy = material.clone();
    if (__LIFT__ && copy.metalness >= 0.5) {
      copy.color.lerp(new THREE.Color(0xffffff), 0.12);
      copy.metalness = Math.min(copy.metalness, 0.55);
      copy.roughness = Math.max(0.18, Math.min(copy.roughness, 0.45));
    }
    if (__DEFINE__) {
      if (copy.transmission > 0.5) {
        copy.transmission *= 0.82;
        copy.roughness = Math.max(copy.roughness, 0.08);
      }
      if (copy.transparent && copy.opacity < 0.85) copy.opacity = Math.min(0.85, copy.opacity + 0.15);
    }
    copies.set(material, copy);
    return copy;
  }
  root.traverse(node => {
    if (node.material) node.material = Array.isArray(node.material) ? node.material.map(adjusted) : adjusted(node.material);
  });
  return root;
}
"""


def material_variant(source, *, lift=False, define=False):
    match = re.search(r"(?m)^[ \t]*export\s+default\s+", source)
    if match is None:
        raise ValueError("Missing default export")
    prefix, body = source[: match.start()], source[match.end() :]
    result = (
        MATERIAL_WRAPPER.replace("__SOURCE__", body)
        if not prefix
        else MATERIAL_WRAPPER.replace("  const build = __SOURCE__;", prefix + "\n  const build = " + body + ";")
    )
    return result.replace("__LIFT__", str(lift).lower()).replace("__DEFINE__", str(define).lower())


class MaterialFinisher(EfficientPipeline):
    async def finish(self, parent):
        candidates = [parent]
        for lift, define in ((True, False), (False, True), (True, True)):
            candidate, _ = await self.accept(
                material_variant(parent.source, lift=lift, define=define), f"material-{lift}-{define}"
            )
            if candidate is not None and all(candidate.front != old.front for old in candidates):
                candidates.append(candidate)
        try:
            selected = await self.select(candidates)
            if selected is not parent and await self.guard(parent, selected):
                return selected
        except BudgetStop:
            pass
        return parent
