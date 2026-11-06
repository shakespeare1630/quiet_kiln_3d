"""Optional final normalization with transformed vertex bounds.

This is an independent static geometry transform. It does not consult benchmark
data or a judge, and keeps special geometry on the conservative fallback path.
It is not enabled in the frozen v0 generator.
"""

import re

WRAPPER = """export default function generate(THREE) {
  // crown-vertex-fit-v1
  const build = __SOURCE__;
  const original = build(THREE);
  original.updateMatrixWorld(true);
  let result = original;
  let compatible = true;
  original.traverse(node => {
    if (node.isInstancedMesh || node.isSkinnedMesh || node.morphTargetInfluences || node.isSprite || node.isLOD)
      compatible = false;
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    if (materials.some(material => material && (material.isShaderMaterial || material.isNodeMaterial)))
      compatible = false;
  });
  if (compatible) {
    try {
      const flattened = new THREE.Group();
      original.traverse(node => {
        if (!node.geometry || !node.geometry.attributes.position) return;
        const copy = node.clone(false);
        copy.geometry = node.geometry.clone();
        copy.geometry.applyMatrix4(node.matrixWorld);
        if (node.isMesh && node.matrixWorld.determinant() < 0) {
          if (!copy.geometry.index) {
            const indices = [];
            for (let i = 0; i < copy.geometry.attributes.position.count; i++) indices.push(i);
            copy.geometry.setIndex(indices);
          }
          const index = copy.geometry.index;
          for (let i = 0; i + 2 < index.count; i += 3) {
            const first = index.getX(i);
            index.setX(i, index.getX(i + 2));
            index.setX(i + 2, first);
          }
          index.needsUpdate = true;
        }
        copy.position.set(0, 0, 0);
        copy.quaternion.set(0, 0, 0, 1);
        copy.scale.set(1, 1, 1);
        copy.matrixAutoUpdate = true;
        for (let ancestor = node; ancestor; ancestor = ancestor.parent) {
          if (!ancestor.visible) copy.visible = false;
        }
        copy.updateMatrix();
        flattened.add(copy);
      });
      if (flattened.children.length) result = flattened;
    } catch (error) { result = original; }
  }
  const root = new THREE.Group();
  root.add(result);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (!(extent > 0) || !Number.isFinite(extent)) throw new Error('Invalid final bounds');
  const center = box.getCenter(new THREE.Vector3());
  const scale = 0.98 / extent;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  root.updateMatrixWorld(true);
  return root;
}
"""


def fit(source: str) -> str:
    if source.startswith("export default function generate(THREE) {\n  // crown-vertex-fit-v1\n"):
        return source.rstrip() + "\n"
    match = re.search(r"(?m)^[ \t]*export\s+default\s+", source)
    if match is None:
        raise ValueError("Missing default export")
    prefix, body = source[: match.start()], source[match.end() :]
    return (
        WRAPPER.replace("__SOURCE__", body)
        if not prefix
        else WRAPPER.replace("  const build = __SOURCE__;", prefix + "\n  const build = " + body + ";")
    )
