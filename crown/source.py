import re

WRAPPER = """export default function generate(THREE) {
  // crown-normalized-v1
  const build = __SOURCE__
  ;
  const root = new THREE.Group();
  root.add(build(THREE));
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (!(extent > 0) || !Number.isFinite(extent)) throw new Error('Invalid geometry bounds');
  const center = box.getCenter(new THREE.Vector3());
  const scale = 0.94 / extent;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  root.updateMatrixWorld(true);
  return root;
}
"""


def extract(reply: str) -> str:
    fenced = re.findall(r"```(?:javascript|js)?\s*\n(.*?)```", reply, re.S)
    source = fenced[-1].strip() if fenced else reply.strip()
    # The full module is deliberately required. Broken/truncated code is sent to
    # the real validator; this regex is not a JavaScript conformance validator.
    if not re.search(r"(?m)^[ \t]*export\s+default\s+function\s+(?:generate\s*)?\(\s*THREE\s*\)", source):
        raise ValueError("Expected one complete export default function generate(THREE)")
    if len(source.encode()) > 950_000:
        raise ValueError("Candidate source too large")
    return source


def normalize(source: str) -> str:
    if source.startswith("export default function generate(THREE) {\n  // crown-normalized-v1\n"):
        return source.rstrip() + "\n"
    exported = re.search(r"(?m)^[ \t]*export\s+default\s+", source)
    if exported is None:
        raise ValueError("Missing default export")
    prefix, body = source[: exported.start()], source[exported.end() :]
    if not prefix:
        return WRAPPER.replace("__SOURCE__", body)
    return WRAPPER.replace("  const build = __SOURCE__", prefix + "\n  const build = " + body)
