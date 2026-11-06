"""Final framing for funnel modules: flat-object tilt, baked transforms, 0.99 fit.

The validator measures the asset with ``Box3.setFromObject``, which wraps each
part's own box after its transform. Rotated parts therefore inflate the box and a
plain fit shrinks the whole object. Moving every mesh transform into its vertices
makes that box tight, so the same model renders larger and still validates.
Instanced meshes keep their per-instance boxes, which the fit measures the same
way as the validator. Very heavy scenes skip the bake to stay within the time limit.
"""

import re

EXPORT = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\(")
MARKER = "// crown-funnel-frame-v1"

FRAME = """
// crown-funnel-frame-v1
function __crownBake(THREE, holder) {
  holder.updateMatrixWorld(true);
  const parts = [];
  let vertices = 0;
  holder.traverse(function (node) {
    if ((node.isMesh || node.isLine || node.isPoints) && node.geometry && node.geometry.attributes.position) {
      parts.push(node);
      vertices += node.geometry.attributes.position.count;
    }
  });
  if (vertices > 250000 || parts.length === 0) return holder;
  const flat = new THREE.Group();
  const worlds = parts.map(function (node) { return node.matrixWorld.clone(); });
  const visible = parts.map(function (node) {
    for (let current = node; current; current = current.parent) if (current.visible === false) return false;
    return true;
  });
  const instance = new THREE.Matrix4();
  for (let p = 0; p < parts.length; p++) {
    const node = parts[p];
    const world = worlds[p];
    if (node.isInstancedMesh) {
      for (let i = 0; i < node.count; i++) {
        node.getMatrixAt(i, instance);
        node.setMatrixAt(i, instance.premultiply(world));
      }
      node.instanceMatrix.needsUpdate = true;
      node.boundingBox = null;
      node.boundingSphere = null;
    } else {
      const geometry = node.geometry.clone();
      geometry.applyMatrix4(world);
      if (node.isMesh && world.determinant() < 0) {
        const count = geometry.index ? geometry.index.count : geometry.attributes.position.count;
        const order = new Uint32Array(count - (count % 3));
        for (let i = 0; i < order.length; i += 3) {
          const a = geometry.index ? geometry.index.getX(i) : i;
          const b = geometry.index ? geometry.index.getX(i + 1) : i + 1;
          const c = geometry.index ? geometry.index.getX(i + 2) : i + 2;
          order[i] = a;
          order[i + 1] = c;
          order[i + 2] = b;
        }
        geometry.setIndex(new THREE.BufferAttribute(order, 1));
      }
      node.geometry = geometry;
    }
  }
  for (let p = 0; p < parts.length; p++) {
    const node = parts[p];
    node.removeFromParent();
    node.position.set(0, 0, 0);
    node.quaternion.identity();
    node.scale.set(1, 1, 1);
    node.visible = visible[p];
    flat.add(node);
  }
  for (let p = 0; p < parts.length; p++) {
    const node = parts[p];
    for (let c = node.children.length - 1; c >= 0; c--) {
      if (parts.indexOf(node.children[c]) < 0) node.remove(node.children[c]);
    }
  }
  flat.updateMatrixWorld(true);
  return flat;
}

export default function generate(THREE) {
  const holder = new THREE.Group();
  holder.add(__crownBuild(THREE));
  holder.updateMatrixWorld(true);
  const shape = new THREE.Box3().setFromObject(holder, true).getSize(new THREE.Vector3());
  if (shape.y < 0.5 * Math.max(shape.x, shape.z)) holder.rotation.x = 15 * Math.PI / 180;
  holder.updateMatrixWorld(true);
  const root = __crownBake(THREE, holder);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (!(extent > 0) || !Number.isFinite(extent)) throw new Error("The asset has no measurable size");
  const center = box.getCenter(new THREE.Vector3());
  const scale = 0.99 / extent;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  root.updateMatrixWorld(true);
  return root;
}
"""


def extract(reply: str) -> str | None:
    """The module in a model reply: the last fenced block that exports a function, else the bare reply."""
    text = (reply or "").strip()
    blocks = [b for b in re.findall(r"```(?:javascript|js|mjs)?[^\n]*\n([\s\S]*?)```", text) if EXPORT.search(b)]
    if blocks:
        text = blocks[-1].strip()
    if not EXPORT.search(text) or MARKER in text:
        return None
    if len(text.encode()) > 900_000:
        return None
    return text + "\n"


def frame(raw: str) -> str:
    """Rename the model's default export and append the framing wrapper."""
    match = EXPORT.search(raw)
    if match is None:
        raise ValueError("The module has no default exported function")
    return raw[: match.start()] + "function __crownBuild(" + raw[match.end() :].rstrip() + "\n" + FRAME
