"""Funnel pipeline: many drafts narrowed by judge-style selection, then rounds of rewrites.

Per prompt:
  1. Drafts: independent modules from the photo, generated concurrently; rejected modules get one repair.
  2. Selection: the selector scores all candidates twice, in opposite presentation orders, on close-ups of the
     judge's four Stage 1 views and the Stage 2 grid in the judge's current neutral light; the best few survive.
  3. Rewrites: every survivor is rewritten from the photo, its own renders at the judge's resolution and its code.
     Parents and children are then selected together, with a small handicap on parents so a rewrite that is at
     least as good replaces its parent.
  4. Framing: the result is tilted if flat, its transforms are baked into vertices and it is fitted to 0.99.

Every model call has an enforced output cap, and a stage starts only if the worst case of all its calls fits the
prompt budget, so the 60-call / 5M-credit limits hold by construction. Request IDs, presentation orders and render
order are fixed per prompt, so a replay with recorded responses makes the same decisions.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal

import numpy as np
import openai
from PIL import Image
from witness_client import RenderOptions, RenderRejected, RenderUnavailable

from . import funnel_prompts as P
from . import media, palette
from .budget import Budget, BudgetStop, EvidenceUnavailable, credits
from .funnel_wrap import extract, frame

LOG = logging.getLogger(__name__)

SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"
# Enforced provider output caps (reasoning included), several times the largest observed module replies.
OUTPUT_CAPS = {(SOL, "high"): 20_000, (SOL, "xhigh"): 32_000, (ASTRA, "high"): 16_000, (SOL, "medium"): 4_000}
IMAGE_TOKENS = 1_800  # upper bound per image; observed about 1,000-1,350
TEXT_BYTES_PER_TOKEN = 2.5  # code and prose tokenize at about 3.5 bytes per token
FRAMING_TOKENS = 2_000

FRONT = RenderOptions(
    thetas=(30.0, 330.0, 0.0, 0.0),
    phis=(0.0, 0.0, 15.0, -30.0),
    img_size=1024,
    cam_radius=2.0,
    cam_fov_deg=49.1,
    gap=0,
    bg_color="ffffff",
    lighting="neutral",
)
GRID = RenderOptions(lighting="neutral")  # the judge's Stage 2 grid in its current neutral light
VIEW_NAMES = ("30 degrees left", "30 degrees right", "from slightly below", "from 30 degrees above")

DRAFT_ID, REDRAFT_ID, REPAIR_OFFSET, SELECT_ID = 1, 50, 500, 9000
LINEAGE_OFFSET, FINAL_SELECT_ID = 10_000, 30_000  # twin: lineage i uses IDs + i * LINEAGE_OFFSET
GUARD_SELECT_ID = 25_000  # checkpoint guard selection (+ id_offset)
JOIN_ATTEMPTS = 6
RENDER_WAITS = (5, 10, 20, 40, 60, 60)  # a busy Pod renderer: wait it out rather than lose the prompt
RENDER_TIMEOUT_RETRIES = 2  # a module near the time limit can pass on one run and fail on the next
LIGHTER = (
    "RENDER_TIMEOUT: the renderer could not finish this module in time. Make it lighter: fewer meshes, fewer "
    "vertices, lower segment counts, smaller textures."
)


@dataclass(frozen=True)
class Round:
    rewrites: tuple[tuple[str, str], ...]  # (model, effort) of each rewrite of every survivor
    keep: int
    parent_margin: float | None = None  # overrides the plan's; negative = a child must beat its parent by that much


@dataclass(frozen=True)
class FunnelPlan:
    drafts: tuple[tuple[str, str, str], ...] = (
        ("plain", SOL, "high"),
        ("proportions", SOL, "high"),
        ("inventory", SOL, "high"),
        ("plain", SOL, "high"),
        ("proportions", SOL, "high"),
        ("inventory", SOL, "high"),
    )
    draft_keep: int = 3
    rounds: tuple[Round, ...] = (
        Round(((SOL, "high"),), keep=2),
        Round(((ASTRA, "high"),), keep=1),
        Round(((SOL, "high"), (SOL, "xhigh")), keep=1),
        Round(((SOL, "xhigh"),), keep=1),
        Round(((SOL, "xhigh"),), keep=1),
    )
    parent_margin: float = 0.25
    grid_weight: float = 0.25
    broken_penalty: float = 2.0
    max_credits: int = 4_900_000
    max_calls: int = 60
    viewpoint: bool = False  # explicit instruction to reproduce the photo's viewing angle (yaw/pitch)
    select_votes: int = 2  # selection calls per decision (mirrored pairs with independent label shuffles)
    critique: bool = False  # the selector also lists fixes; the next rewrite of a survivor receives them
    palette: bool = False  # measured photo colours in every prompt; rewrites also get the current module's colours
    checkpoints: tuple[int, ...] = ()  # rounds whose champions re-compete with the last one in a final guard selection


DEFAULT_PLAN = FunnelPlan()
# v2: wider early search (8 drafts, two rewrites per survivor in round 1); later rounds unchanged.
WIDE_PLAN = FunnelPlan(
    drafts=(
        ("plain", SOL, "high"),
        ("proportions", SOL, "high"),
        ("inventory", SOL, "high"),
        ("plain", SOL, "high"),
        ("proportions", SOL, "high"),
        ("inventory", SOL, "high"),
        ("plain", SOL, "high"),
        ("proportions", SOL, "high"),
    ),
    rounds=(
        Round(((SOL, "high"), (SOL, "high")), keep=2),
        Round(((ASTRA, "high"),), keep=1),
        Round(((SOL, "high"), (SOL, "xhigh")), keep=1),
        Round(((SOL, "xhigh"),), keep=1),
        Round(((SOL, "xhigh"),), keep=1),
    ),
)
VIEWPOINT_PLAN = FunnelPlan(viewpoint=True)
# plus: wide early search and four selection votes per decision (lower selection noise).
PLUS_PLAN = FunnelPlan(drafts=WIDE_PLAN.drafts, rounds=WIDE_PLAN.rounds, select_votes=4)
CRITIQUE_PLAN = FunnelPlan(critique=True)
# deep: F1 plus three more single-lineage rewrite rounds (the selector agrees with the judge on late rounds,
# and children keep improving), using budget F1 leaves unspent; the stage guard still caps at 4.9M / 60 calls.
DEEP_PLAN = FunnelPlan(
    rounds=DEFAULT_PLAN.rounds
    + (
        Round(((SOL, "high"),), keep=1),
        Round(((SOL, "xhigh"),), keep=1),
        Round(((SOL, "high"),), keep=1),
    )
)
# beam: F1 trace stats show Sol-high rewrites beat Sol-xhigh head-to-head in round 3 (15 vs 7 wins) and single
# late rewrites beat their parent ~71% of the time, so late rounds use two Sol-high children each, one round more.
BEAM_PLAN = FunnelPlan(
    rounds=(
        Round(((SOL, "high"),), keep=2),
        Round(((ASTRA, "high"),), keep=1),
        Round(((SOL, "high"), (SOL, "high")), keep=1),
        Round(((SOL, "high"), (SOL, "high")), keep=1),
        Round(((SOL, "high"), (SOL, "high")), keep=1),
        Round(((SOL, "high"), (SOL, "high")), keep=1),
    )
)
# lite: F1's shape with every rewrite on Sol high. Per-call costs in F1 recordings: Sol high ~117k credits, Sol
# xhigh ~239k, Astra high ~522k, and neither beat Sol high per call (round-3 head-to-head Sol high 15 vs xhigh 7),
# so this keeps the search and roughly halves the cost (~2.0M vs ~3.3M credits per prompt).
LITE_PLAN = FunnelPlan(
    rounds=(
        Round(((SOL, "high"),), keep=2),
        Round(((SOL, "high"),), keep=1),
        Round(((SOL, "high"), (SOL, "high")), keep=1),
        Round(((SOL, "high"),), keep=1),
        Round(((SOL, "high"),), keep=1),
    )
)
# lite-deep: lite + ten more single-child Sol-high rounds (15 rounds, ~57 calls, ~3.8M credits). On F1's harvest
# the judged net rose from +7.5% after round 3 to +25.0% after round 5 (20 prompts), with no plateau in sight.
# lite12: lite with 12 drafts (the first 6 identical to lite's). The top-ranked draft drives the final (shared-twin
# decomposition), and drafts are cheap (~0.1M) and parallel, so spend budget on finding a better starting point.
LITE12_PLAN = FunnelPlan(drafts=FunnelPlan().drafts * 2, rounds=LITE_PLAN.rounds)
# lite-astra: lite's six Sol drafts plus two Astra drafts (8-way draft selection). The top draft drives the final;
# test whether the stronger, 5x-priced model drafts better starting points (~+1.0M credits).
LITE_ASTRA_PLAN = FunnelPlan(
    drafts=FunnelPlan().drafts + (("plain", ASTRA, "high"), ("inventory", ASTRA, "high")), rounds=LITE_PLAN.rounds
)
# twin4: twin with 4 drafts per lineage so both independent lineages (each seeded from its own top draft) finish all
# lite rounds under the per-lineage cap. Shared-draft twin showed the top-ranked draft drives the final (+1.80 per
# 4 duels for the lineage seeded with ranks 1,3 vs +0.70 for ranks 2,4), and separate pools keep finals diverse
# enough for the final pick (74% vs 62% accuracy).
TWIN4_PLAN = FunnelPlan(
    drafts=(("plain", SOL, "high"), ("proportions", SOL, "high"), ("inventory", SOL, "high"), ("plain", SOL, "high")),
    rounds=LITE_PLAN.rounds,
)
LITE_DEEP_PLAN = FunnelPlan(rounds=LITE_PLAN.rounds + tuple(Round(((SOL, "high"),), keep=1) for _ in range(10)))
# deep rounds accepted many children on near-ties (parent handicap 0.25) and drifted on some prompts; guard = final
# selection among the round-5, round-10 and last champions (2 more calls, 59 total); strict = from round 6 a child
# must beat its parent by 0.5 in selector score, plus the guard.
LITE_DEEP_GUARD_PLAN = FunnelPlan(rounds=LITE_DEEP_PLAN.rounds, checkpoints=(5, 10))
LITE_DEEP_STRICT_PLAN = FunnelPlan(
    rounds=LITE_PLAN.rounds + tuple(Round(((SOL, "high"),), keep=1, parent_margin=-0.5) for _ in range(10)),
    checkpoints=(5, 10),
)
# duo: per-prompt outcomes swing a lot between runs while the selector agrees with the judge on ~97% of decided
# pairs, so two all-Sol-high lineages stay alive to the end (keep 2 every round) and the final selection picks one.
DUO_PLAN = FunnelPlan(rounds=tuple(Round(((SOL, "high"),), keep=2) for _ in range(5)))
# palette: F1 + measured photo colours (all prompts) and the current module's measured colours (rewrites).
PALETTE_PLAN = FunnelPlan(palette=True)
LITE_PALETTE_PLAN = FunnelPlan(rounds=LITE_PLAN.rounds, palette=True)
PLANS = {
    "funnel": DEFAULT_PLAN,
    "funnel-wide": WIDE_PLAN,
    "funnel-view": VIEWPOINT_PLAN,
    "funnel-plus": PLUS_PLAN,
    "funnel-critique": CRITIQUE_PLAN,
    "funnel-deep": DEEP_PLAN,
    "funnel-beam": BEAM_PLAN,
    "funnel-duo": DUO_PLAN,
    "funnel-lite": LITE_PLAN,
    "funnel-palette": PALETTE_PLAN,
    "funnel-lite-palette": LITE_PALETTE_PLAN,
    "funnel-twin": LITE_PLAN,
    "funnel-lite-deep": LITE_DEEP_PLAN,
    "funnel-lite-deep-guard": LITE_DEEP_GUARD_PLAN,
    "funnel-lite-deep-strict": LITE_DEEP_STRICT_PLAN,
    "funnel-shared-twin": LITE_PLAN,  # SharedTwinPipeline over lite rounds
    "funnel-twin4": TWIN4_PLAN,  # TwinPipeline, 4 drafts per lineage
    "funnel-lite12": LITE12_PLAN,
    "funnel-seq-twin": LITE_PLAN,  # SequentialTwinPipeline over lite
    "funnel-lite-astra": LITE_ASTRA_PLAN,  # TwinPipeline over two lite lineages (see TwinPipeline)
}


@dataclass
class Candidate:
    name: str
    raw: str
    source: str
    views: list[bytes]
    sheet: bytes
    grid: bytes
    parent: "Candidate | None" = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)  # fixes from the latest selection that scored it
    colours: str = ""  # measured judge-view colours and overlap with the photo (palette plans only)


def jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def image_item(data: bytes) -> dict:
    kind = "jpeg" if data.startswith(b"\xff\xd8") else "png"
    return {
        "type": "input_image",
        "image_url": f"data:image/{kind};base64," + base64.b64encode(data).decode(),
        "detail": "high",
    }


def object_closeup(photo: bytes) -> bytes | None:
    """The photo cropped to its object, or None when the object already fills most of it or cannot be found."""
    image = Image.open(io.BytesIO(photo)).convert("RGB")
    small = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    mask = np.sqrt(((small - np.median(border, axis=0)) ** 2).sum(axis=2)) > 30.0
    if not 0.002 < mask.mean() < 0.9:
        return None
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    pad = 0.08 * max(int(xs[-1] - xs[0]), int(ys[-1] - ys[0]))
    left, right = max(0.0, xs[0] - pad) / 256, min(256.0, xs[-1] + 1 + pad) / 256
    top, bottom = max(0.0, ys[0] - pad) / 256, min(256.0, ys[-1] + 1 + pad) / 256
    if (right - left) * (bottom - top) > 0.6:
        return None
    crop = image.crop(
        (int(left * image.width), int(top * image.height), int(right * image.width), int(bottom * image.height))
    )
    scale = 1024 / max(crop.size)
    return jpeg(
        crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    )


def view_closeups(front: bytes, output: int = 640) -> list[bytes]:
    """The four tiles of a gap-free 2x2 render on white, cropped around the object at one common scale."""
    image = Image.open(io.BytesIO(front)).convert("RGB")
    pixels = np.asarray(image)
    width, height = image.width // 2, image.height // 2
    boxes = []
    for index in range(4):
        top, left = (index // 2) * height, (index % 2) * width
        mask = (pixels[top : top + height, left : left + width] < 250).any(axis=2)
        ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        boxes.append((left + xs[0], top + ys[0], left + xs[-1] + 1, top + ys[-1] + 1) if ys.size else None)
    extents = [max(b[2] - b[0], b[3] - b[1]) for b in boxes if b is not None]
    limit = min(width, height)
    side = min(limit, max(limit // 4, int(max(extents, default=limit) * 1.12) + 8))
    views = []
    for index, box in enumerate(boxes):
        top, left = (index // 2) * height, (index % 2) * width
        cx = (box[0] + box[2]) // 2 if box else left + width // 2
        cy = (box[1] + box[3]) // 2 if box else top + height // 2
        x0 = min(max(cx - side // 2, left), left + width - side)
        y0 = min(max(cy - side // 2, top), top + height - side)
        views.append(
            jpeg(image.crop((x0, y0, x0 + side, y0 + side)).resize((output, output), Image.Resampling.LANCZOS))
        )
    return views


def view_sheet(views: list[bytes], tile: int = 512, gap: int = 8) -> bytes:
    sheet = Image.new("RGB", (2 * tile + gap, 2 * tile + gap), (200, 200, 200))
    for index, view in enumerate(views):
        image = Image.open(io.BytesIO(view)).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        sheet.paste(image, ((index % 2) * (tile + gap), (index // 2) * (tile + gap)))
    return jpeg(sheet)


def fallback_module(photo: bytes) -> str:
    """A last-resort blob in the photo's central color: a delivered asset can still draw an empty opponent."""
    image = Image.open(io.BytesIO(photo)).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    red, green, blue = (int(v) for v in np.median(np.asarray(image)[16:48, 16:48].reshape(-1, 3), axis=0))
    return frame(
        "export default function generate(THREE) {\n"
        f"  const material = new THREE.MeshStandardMaterial({{ color: 0x{red:02x}{green:02x}{blue:02x}, roughness: 0.6 }});\n"
        "  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.5, 48, 32), material);\n"
        "  mesh.scale.set(1, 0.85, 1);\n"
        "  return mesh;\n"
        "}\n"
    )


def reservation(model: str, effort: str, instructions: str, content: list[dict]):
    text_bytes = len(instructions.encode()) + sum(len(item.get("text", "").encode()) for item in content)
    images = sum(item["type"] == "input_image" for item in content)
    input_tokens = math.ceil(text_bytes / TEXT_BYTES_PER_TOKEN) + IMAGE_TOKENS * images + FRAMING_TOKENS
    return credits(model, {"input_tokens": input_tokens, "output_tokens": OUTPUT_CAPS[(model, effort)]})


@dataclass
class Spec:
    request_id: int
    model: str
    effort: str
    instructions: str
    content: list[dict]
    schema: dict | None = None


@dataclass
class Ledger:
    """Worst-case credits and calls of stages in flight; lets concurrent lineages share one budget safely."""

    credits: Decimal = field(default_factory=Decimal)
    calls: int = 0


class FunnelPipeline:
    def __init__(
        self,
        image: bytes,
        seed: int,
        witness,
        config=None,
        plan: FunnelPlan = DEFAULT_PLAN,
        *,
        id_offset: int = 0,
        budget: Budget | None = None,
        ledger: Ledger | None = None,
    ):
        self.photo = media.canonical(image)
        self.seed = seed
        self.witness = witness
        self.plan = plan
        self.id_offset = id_offset
        self.budget = budget or Budget(plan.max_credits, plan.max_calls)
        self.ledger = ledger or Ledger()
        self.digest = hashlib.sha256(image).hexdigest()
        closeup = object_closeup(self.photo)
        self.target = [media.text(P.TARGET), image_item(self.photo)]
        if closeup is not None:
            self.target += [media.text(P.CLOSEUP), image_item(closeup)]
        self.photo_pixels = palette.photo_pixels(self.photo) if plan.palette else None
        # brief = what code writers see (target + measured palette); the selector sees only the target.
        self.brief = list(self.target)
        if self.photo_pixels is not None:
            colours = palette.describe(palette.palette(self.photo_pixels))
            self.brief.append(media.text(P.PALETTE.format(colours=colours)))
        self.best_so_far: str | None = None
        self.trace: list[dict] = []
        self.generate_text = P.GENERATE_VIEWPOINT if plan.viewpoint else P.GENERATE
        self.rewrite_text = P.REWRITE_VIEWPOINT if plan.viewpoint else P.REWRITE

    def describe(self) -> dict:
        return {
            "pipeline": "funnel-v1",
            **asdict(self.plan),
            "output_caps": {f"{m}/{e}": c for (m, e), c in OUTPUT_CAPS.items()},
        }

    # ----- model calls -----

    async def call(self, spec: Spec) -> str | None:
        params = dict(
            request_id=spec.request_id,
            model=spec.model,
            instructions=spec.instructions,
            input=[{"role": "user", "content": spec.content}],
            reasoning={"effort": spec.effort},
            max_output_tokens=OUTPUT_CAPS[(spec.model, spec.effort)],
            service_tier="default",
            store=False,
        )
        if spec.schema is not None:
            params["text"] = spec.schema
        for attempt in range(JOIN_ATTEMPTS):
            try:
                response = await self.witness.responses.create(**params)
                break
            except openai.APITimeoutError:
                # A capture that outlasted the client's wait: repeating the same request joins it without paying again.
                if attempt == JOIN_ATTEMPTS - 1:
                    raise
            except openai.RateLimitError:
                raise  # leave the prompt for a later pass; saved calls are reused
            except openai.APIStatusError as error:
                # A recorded provider failure: it counts as a call and is never retried under the same ID.
                LOG.warning("request=%d provider error %s", spec.request_id, error.status_code)
                self.budget.settle(spec.request_id, spec.model, {"input_tokens": 0, "output_tokens": 0})
                return None
            except openai.APIConnectionError:
                raise  # transport failure: the runner's later pass recovers the saved exchange
            except openai.APIError as error:
                # An error event inside a delivered stream (e.g. "servers are currently overloaded"): the exchange is
                # recorded as failed, so a later pass would see the same failure. Drop this one candidate instead.
                LOG.warning("request=%d provider stream error: %s", spec.request_id, str(error)[:120])
                self.budget.settle(spec.request_id, spec.model, {"input_tokens": 0, "output_tokens": 0})
                return None
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            raise EvidenceUnavailable("The captured response has no usage")
        self.budget.settle(spec.request_id, spec.model, dict(usage))
        if getattr(response, "status", None) != "completed":
            LOG.info("request=%d incomplete", spec.request_id)
            return None
        text = getattr(response, "output_text", None)
        return text if isinstance(text, str) else None

    async def stage(self, specs: list[Spec]) -> list[str | None]:
        """Admit the longest prefix of calls whose worst case fits, then run them concurrently."""
        admitted = list(specs)
        total = 0
        while admitted:
            try:
                total = sum(reservation(s.model, s.effort, s.instructions, s.content) for s in admitted)
                self.budget.check(total + self.ledger.credits, calls=len(admitted) + self.ledger.calls)
                break
            except BudgetStop:
                admitted.pop()
        if not admitted:
            total = 0
        # Check and reservation happen without an await in between, so concurrent lineages cannot both claim it.
        self.ledger.credits += total
        self.ledger.calls += len(admitted)
        try:
            replies = list(await asyncio.gather(*(self.call(spec) for spec in admitted))) if admitted else []
        finally:
            self.ledger.credits -= total
            self.ledger.calls -= len(admitted)
        LOG.info("stage calls=%d/%d spent=%s", len(admitted), len(specs), self.budget.spent)
        return replies + [None] * (len(specs) - len(admitted))

    # ----- candidates -----

    async def render(self, source: str, options: RenderOptions) -> bytes:
        """Render, waiting out a busy renderer; repeated timeouts count as a rejection the model can fix."""
        waits = iter(RENDER_WAITS)
        timeouts = 0
        while True:
            try:
                return await self.witness.render(source, options)
            except RenderRejected as rejection:
                if "TIMEOUT_EXCEEDED" not in str(getattr(rejection, "detail_json", "")):
                    raise
                timeouts += 1
                if timeouts > RENDER_TIMEOUT_RETRIES:
                    raise
            except RenderUnavailable as error:
                response = getattr(error.__cause__, "response", None)
                if getattr(response, "status_code", None) == 500:
                    timeouts += 1
                    if timeouts > RENDER_TIMEOUT_RETRIES:
                        raise RenderRejected(json.dumps({"error": LIGHTER})) from error
                wait = next(waits, None)
                if wait is None:
                    raise
                await asyncio.sleep(wait)

    async def build(self, name: str, raw: str, parent: Candidate | None) -> Candidate | str:
        source = frame(raw)
        try:
            front = await self.render(source, FRONT)
            grid = await self.render(source, GRID)
        except RenderRejected as rejection:
            return str(getattr(rejection, "detail_json", rejection))[:2500]
        views = view_closeups(front)
        candidate = Candidate(name, raw, source, views, view_sheet(views), jpeg(Image.open(io.BytesIO(grid))), parent)
        if self.photo_pixels is not None:
            pixels = palette.render_pixels(front)
            candidate.colours = P.CURRENT_COLOURS.format(
                colours=palette.describe(palette.palette(pixels)), overlap=palette.overlap(self.photo_pixels, pixels)
            )
        return candidate

    async def materialize(self, items: list[tuple[str, Spec, str | None, Candidate | None]]) -> list[Candidate | None]:
        """Render replies in order; each rejected module gets one repair, rendered afterwards in the same order."""
        out: list[Candidate | None] = [None] * len(items)
        rejected = []
        for index, (name, spec, reply, parent) in enumerate(items):
            raw = extract(reply) if reply else None
            if raw is None:
                continue
            built = await self.build(name, raw, parent)
            if isinstance(built, Candidate):
                out[index] = built
            else:
                rejected.append((index, raw, built))
        if rejected:
            repairs = [
                Spec(
                    items[index][1].request_id + REPAIR_OFFSET,
                    SOL,
                    "high",
                    self.generate_text,
                    [*self.brief, media.text(P.REPAIR.format(errors=errors, source=raw.rstrip()))],
                )
                for index, raw, errors in rejected
            ]
            for (index, _, _), reply in zip(rejected, await self.stage(repairs), strict=True):
                raw = extract(reply) if reply else None
                if raw is not None:
                    built = await self.build(items[index][0] + "r", raw, items[index][3])
                    if isinstance(built, Candidate):
                        out[index] = built
        return out

    # ----- selection -----

    def select_content(self, labels: dict[str, Candidate], order: list[str]) -> list[dict]:
        content = [media.text("TARGET:"), *self.target[1:]]
        for label in order:
            content += [
                media.text(f"CANDIDATE {label} VIEWS:"),
                image_item(labels[label].sheet),
                media.text(f"CANDIDATE {label} GRID:"),
                image_item(labels[label].grid),
            ]
        return content

    def scores(self, labels: dict[str, Candidate], replies: list[str | None]) -> dict[str, float]:
        collected: dict[str, list[float]] = {}
        for reply in replies:
            try:
                entries = json.loads(reply or "")["candidates"]
            except (ValueError, KeyError, TypeError):
                continue
            seen = set()
            for entry in entries if isinstance(entries, list) else []:
                label = entry.get("label") if isinstance(entry, dict) else None
                if label not in labels or label in seen:
                    continue
                try:
                    fronts = [min(10, max(0, int(entry[k]))) for k in P.FIELDS[:4]]
                    grid = min(10, max(0, int(entry["grid"])))
                    broken = int(entry["broken_sides"])
                except (KeyError, TypeError, ValueError):
                    continue
                seen.add(label)
                if self.plan.critique and isinstance(entry.get("fixes"), list):
                    notes = labels[label].notes
                    for fix in entry["fixes"][:5]:
                        text = " ".join(str(fix).split())[:200]
                        if text and text not in notes and len(notes) < 8:
                            notes.append(text)
                score = sum(fronts) / 4 + self.plan.grid_weight * grid
                if broken >= 3:
                    score += self.plan.broken_penalty
                collected.setdefault(label, []).append(score)
        return {labels[label].name: sum(values) / len(values) for label, values in collected.items()}

    async def select(
        self, candidates: list[Candidate], keep: int, parents: set[str], request_id: int, margin: float | None = None
    ) -> list[Candidate]:
        """The best `keep` candidates; list children before parents, since ties keep list order."""
        if len(candidates) <= keep:
            return candidates
        specs, mappings = [], []
        instructions = P.SELECT_CRITIQUE if self.plan.critique else P.SELECT
        if self.plan.critique:
            for candidate in candidates:
                candidate.notes = []
        for vote in range(max(1, self.plan.select_votes // 2)):
            shuffled = candidates.copy()
            salt = "" if vote == 0 else f":{vote}"
            random.Random(f"{self.seed}:{self.digest}:{request_id}{salt}").shuffle(shuffled)
            labels = {chr(65 + i): candidate for i, candidate in enumerate(shuffled)}
            forward = list(labels)
            schema = P.select_schema(forward, fixes=self.plan.critique)
            specs += [
                Spec(request_id + 2 * vote, SOL, "medium", instructions, self.select_content(labels, forward), schema),
                Spec(
                    request_id + 2 * vote + 1,
                    SOL,
                    "medium",
                    instructions,
                    self.select_content(labels, forward[::-1]),
                    schema,
                ),
            ]
            mappings.append(labels)
        replies = await self.stage(specs)
        collected: dict[str, list[float]] = {}
        for vote, labels in enumerate(mappings):
            for name, value in self.scores(labels, replies[2 * vote : 2 * vote + 2]).items():
                collected.setdefault(name, []).append(value)
        scored = {name: sum(values) / len(values) for name, values in collected.items()}

        def rank(item):
            index, candidate = item
            score = scored.get(candidate.name, 99.0)
            handicap = self.plan.parent_margin if margin is None else margin
            return (score + (handicap if candidate.name in parents else 0.0), index)

        ranked = [candidate for _, candidate in sorted(enumerate(candidates), key=rank)]
        self.trace.append({"select": request_id, "scores": scored, "kept": [c.name for c in ranked[:keep]]})
        return ranked[:keep]

    # ----- run -----

    def merge_content(self, a: Candidate, b: Candidate) -> list[dict]:
        content = list(self.brief)
        for label, candidate in (("A", a), ("B", b)):
            for name, view in zip(VIEW_NAMES, candidate.views, strict=True):
                content += [media.text(f"Module {label}, {name}:"), image_item(view)]
            content += [media.text(f"Module {label}, Stage 2 grid on gray:"), image_item(candidate.grid)]
        content.append(media.text(P.MERGE.format(source_a=a.raw.rstrip(), source_b=b.raw.rstrip())))
        return content

    def rewrite_content(self, parent: Candidate) -> list[dict]:
        content = list(self.brief)
        for name, view in zip(VIEW_NAMES, parent.views, strict=True):
            content += [media.text(f"Current module, {name}:"), image_item(view)]
        content += [
            media.text("Current module, Stage 2 grid on gray:"),
            image_item(parent.grid),
            *(
                [
                    media.text(
                        "Notes from a strict comparison of the current module with the photo (verify them against the "
                        "images; the photo wins where they disagree):\n"
                        + "\n".join("- " + note for note in parent.notes)
                    )
                ]
                if self.plan.critique and parent.notes
                else []
            ),
            *([media.text(parent.colours)] if parent.colours else []),
            media.text(self.rewrite_text.format(source=parent.raw.rstrip())),
        ]
        return content

    async def drafts(self, base: int, plan: tuple[tuple[str, str, str], ...]) -> list[Candidate]:
        specs = [
            Spec(
                self.id_offset + base + i,
                model,
                effort,
                self.generate_text,
                [*self.brief, media.text(P.DRAFT_STARTS[start])],
            )
            for i, (start, model, effort) in enumerate(plan)
        ]
        replies = await self.stage(specs)
        items = [(f"d{spec.request_id}", spec, reply, None) for spec, reply in zip(specs, replies, strict=True)]
        return [c for c in await self.materialize(items) if c is not None]

    def rewrite_request_id(self, index: int, position: int, number: int) -> int:
        """Rounds 1-8 keep the original IDs (1000 * round); later rounds move to 100000 + 1000 * index, since round 9
        would reach the selection IDs (9000+). Repairs add REPAIR_OFFSET (500) to these."""
        base = 1000 * (index + 1) if index < 8 else 100_000 + 1000 * index
        return self.id_offset + base + 10 * position + number

    async def advance(self, index: int, survivors: list[Candidate], round_: Round) -> list[Candidate]:
        specs, meta = [], []
        for position, parent in enumerate(survivors):
            for number, (model, effort) in enumerate(round_.rewrites):
                spec = Spec(
                    self.rewrite_request_id(index, position, number),
                    model,
                    effort,
                    self.generate_text,
                    self.rewrite_content(parent),
                )
                specs.append(spec)
                meta.append((f"r{index + 1}.{position}.{number}", spec, parent))
        replies = await self.stage(specs)
        children = await self.materialize([(n, s, r, p) for (n, s, p), r in zip(meta, replies, strict=True)])
        pool = [c for c in children if c is not None] + survivors
        select_id = self.id_offset + SELECT_ID + 10 * (index + 1)
        return await self.select(pool, round_.keep, {p.name for p in survivors}, select_id, round_.parent_margin)

    async def run(self) -> str:
        winner = await self.run_candidate()
        return winner.source if winner is not None else fallback_module(self.photo)

    async def run_candidate(self) -> Candidate | None:
        survivors = await self.drafts(DRAFT_ID, self.plan.drafts)
        if not survivors:
            survivors = await self.drafts(REDRAFT_ID, self.plan.drafts[:3])
        if not survivors:
            return None
        self.best_so_far = survivors[0].source
        survivors = await self.select(survivors, self.plan.draft_keep, set(), self.id_offset + SELECT_ID)
        return await self.run_rounds(survivors)

    async def run_rounds(self, survivors: list[Candidate]) -> Candidate:
        """The plan's rewrite rounds from given survivors (plus the optional checkpoint guard); the lineage winner."""
        self.best_so_far = survivors[0].source
        saved: list[Candidate] = []
        for index, round_ in enumerate(self.plan.rounds):
            survivors = await self.advance(index, survivors, round_)
            self.best_so_far = survivors[0].source
            if index + 1 in self.plan.checkpoints and all(c.name != survivors[0].name for c in saved):
                saved.append(survivors[0])
        winner = survivors[0]
        contenders = [c for c in saved if c.name != winner.name] + [winner]
        if len(contenders) > 1:
            # Late rounds can drift (near-tie acceptances); the accumulated difference is easier to judge.
            winner = (await self.select(contenders[::-1], 1, set(), self.id_offset + GUARD_SELECT_ID))[0]
        self.trace.append({"winner": winner.name, "lineage": self.lineage(winner)})
        return winner

    @staticmethod
    def lineage(candidate: Candidate) -> list[str]:
        names = []
        while candidate is not None:
            names.append(candidate.name)
            candidate = candidate.parent
        return names


FINAL_CREDITS, FINAL_CALLS = 160_000, 2  # final selection's own allowance (2 Sol-medium votes on 2 finals)


class SplitBudget:
    """Read-only union of the lineage and final budgets, for reports (charges, spent)."""

    def __init__(self, parts: list[Budget]):
        self.parts = parts

    @property
    def charges(self) -> dict[int, Decimal]:
        merged: dict[int, Decimal] = {}
        for part in self.parts:
            merged.update(part.charges)
        return merged

    @property
    def spent(self) -> Decimal:
        return sum((part.spent for part in self.parts), Decimal(0))


class TwinPipeline:
    """Independent lineages (own drafts, selections and rewrites), then one final mirrored selection between their
    winners.

    Same-prompt runs swing widely (paired arm differences span -5..+5 duels of 4 opponents), so picking the better
    of two complete runs beats one run whenever the selector can tell them apart (measured: +0.55 net per 4 duels).

    Replay safety: each lineage has its own fixed budget ((max - final allowance) / lineages credits and calls) and
    the final selection its own, so every admission decision depends only on that lineage's recorded responses,
    never on how concurrent calls interleave. (A shared budget made truncation timing-dependent, which an
    official replay could not reproduce.) Request IDs of lineage i are offset by i * LINEAGE_OFFSET.
    """

    def __init__(self, image: bytes, seed: int, witness, config=None, plan: FunnelPlan | None = None, lineages=2):
        self.plan = plan or LITE_PLAN
        credits_each = (self.plan.max_credits - FINAL_CREDITS) // lineages
        calls_each = (self.plan.max_calls - FINAL_CALLS) // lineages
        self.lines = [
            FunnelPipeline(
                image,
                seed,
                witness,
                plan=self.plan,
                id_offset=i * LINEAGE_OFFSET,
                budget=Budget(credits_each, calls_each),
            )
            for i in range(lineages)
        ]
        self.judge = FunnelPipeline(image, seed, witness, plan=self.plan, budget=Budget(FINAL_CREDITS, FINAL_CALLS))
        self.budget = SplitBudget([line.budget for line in self.lines] + [self.judge.budget])
        self.photo = self.lines[0].photo
        self.trace: list[dict] = []

    @property
    def best_so_far(self) -> str | None:
        return next((line.best_so_far for line in self.lines if line.best_so_far), None)

    def describe(self) -> dict:
        return {
            **self.lines[0].describe(),
            "pipeline": "funnel-twin",
            "lineages": len(self.lines),
            "lineage_budget": {
                "credits": int(self.lines[0].budget.max_credits),
                "calls": self.lines[0].budget.max_calls,
            },
            "final_budget": {"credits": FINAL_CREDITS, "calls": FINAL_CALLS},
        }

    async def run(self) -> str:
        winners = await asyncio.gather(*(line.run_candidate() for line in self.lines))
        for index, line in enumerate(self.lines):
            self.trace.append({"lineage": index, "trace": line.trace})
        finals = [replace(w, name=f"L{index}:{w.name}") for index, w in enumerate(winners) if w is not None]
        if not finals:
            return fallback_module(self.photo)
        best = (await self.judge.select(finals, 1, set(), FINAL_SELECT_ID))[0]
        self.trace.append(
            {"final": [c.name for c in finals], "winner": best.name, "select": (self.judge.trace or [{}])[-1]}
        )
        return best.source


class SharedTwinPipeline:
    """One shared draft stage, then independent lineages seeded with interleaved top drafts (lineage i gets ranks
    i, i + n, ...), then a final mirrored selection between the lineage winners.

    Twin with separate drafts ran out of budget: 22 of 60 lineages were cut before round 5, and rounds 4-5 are worth
    +0.70 per prompt (F1 harvest). Sharing the drafts saves ~0.6M credits and 8 calls per extra lineage, so both
    lineages finish all rounds. Replay safety: the draft stage is sequential; lineage budgets are computed from its
    recorded spend and fixed before the lineages start; the final selection has its own allowance.
    IDs: drafts and their selection as in the single funnel; lineage i offset by i * LINEAGE_OFFSET; final 30000.
    """

    def __init__(self, image: bytes, seed: int, witness, config=None, plan: FunnelPlan | None = None, lineages=2):
        self.plan = plan or LITE_PLAN
        self.count = lineages
        self.root = FunnelPipeline(image, seed, witness, plan=self.plan)
        self.image, self.seed, self.witness = image, seed, witness
        self.lines: list[FunnelPipeline] = []
        self.judge = FunnelPipeline(image, seed, witness, plan=self.plan, budget=Budget(FINAL_CREDITS, FINAL_CALLS))
        self.photo = self.root.photo
        self.trace: list[dict] = []

    @property
    def budget(self) -> SplitBudget:
        return SplitBudget([self.root.budget, *(line.budget for line in self.lines), self.judge.budget])

    @property
    def best_so_far(self) -> str | None:
        return next((line.best_so_far for line in self.lines if line.best_so_far), None) or self.root.best_so_far

    def describe(self) -> dict:
        return {**self.root.describe(), "pipeline": "funnel-shared-twin", "lineages": self.count}

    async def run(self) -> str:
        root = self.root
        drafts = await root.drafts(DRAFT_ID, self.plan.drafts)
        if not drafts:
            drafts = await root.drafts(REDRAFT_ID, self.plan.drafts[:3])
        if not drafts:
            return fallback_module(self.photo)
        root.best_so_far = drafts[0].source
        ranked = await root.select(drafts, 2 * self.count, set(), SELECT_ID)
        root.best_so_far = ranked[0].source
        self.trace.append({"drafts": root.trace})
        spent, calls = root.budget.spent, len(root.budget.charges)
        credits_each = int((root.budget.max_credits - spent - FINAL_CREDITS) // self.count)
        calls_each = (root.budget.max_calls - calls - FINAL_CALLS) // self.count
        seeds = [ranked[i :: self.count] for i in range(self.count)]
        self.lines = [
            FunnelPipeline(
                self.image,
                self.seed,
                self.witness,
                plan=self.plan,
                id_offset=i * LINEAGE_OFFSET,
                budget=Budget(credits_each, calls_each),
            )
            for i in range(self.count)
            if seeds[i]
        ]
        winners = await asyncio.gather(*(line.run_rounds(seed) for line, seed in zip(self.lines, seeds, strict=False)))
        for index, line in enumerate(self.lines):
            self.trace.append({"lineage": index, "trace": line.trace})
        finals = [replace(w, name=f"L{index}:{w.name}") for index, w in enumerate(winners) if w is not None]
        if len(finals) == 1:
            return finals[0].source
        best = (await self.judge.select(finals, 1, set(), FINAL_SELECT_ID))[0]
        self.trace.append(
            {"final": [c.name for c in finals], "winner": best.name, "select": (self.judge.trace or [{}])[-1]}
        )
        return best.source


class SequentialTwinPipeline:
    """Lineage 0 is exactly the single lite run (whole budget minus the final allowance, which lite never reaches);
    lineage 1 then runs with what is left; a final mirrored selection picks between the two winners.

    Parallel twins split the budget up front and their lineages came out weaker than a standalone lite run
    (random-lineage mean ~+1.0 vs lite +1.57 per 4 duels over three variants). Here lineage 0 is the floor: the
    result can differ from lite only where the final pick prefers lineage 1. Fully sequential, so replay is
    deterministic; wall time is about twice a lite run.
    """

    def __init__(self, image: bytes, seed: int, witness, config=None, plan: FunnelPlan | None = None):
        self.plan = plan or LITE_PLAN
        self.image, self.seed, self.witness = image, seed, witness
        first = Budget(self.plan.max_credits - FINAL_CREDITS, self.plan.max_calls - FINAL_CALLS)
        self.lines = [FunnelPipeline(image, seed, witness, plan=self.plan, budget=first)]
        self.judge = FunnelPipeline(image, seed, witness, plan=self.plan, budget=Budget(FINAL_CREDITS, FINAL_CALLS))
        self.photo = self.lines[0].photo
        self.trace: list[dict] = []

    @property
    def budget(self) -> SplitBudget:
        return SplitBudget([*(line.budget for line in self.lines), self.judge.budget])

    @property
    def best_so_far(self) -> str | None:
        return next((line.best_so_far for line in self.lines if line.best_so_far), None)

    def describe(self) -> dict:
        return {**self.lines[0].describe(), "pipeline": "funnel-seq-twin", "lineages": 2}

    async def run(self) -> str:
        first = await self.lines[0].run_candidate()
        spent, calls = self.lines[0].budget.spent, len(self.lines[0].budget.charges)
        second_budget = Budget(
            int(self.plan.max_credits - FINAL_CREDITS - spent), self.plan.max_calls - FINAL_CALLS - calls
        )
        second_line = FunnelPipeline(
            self.image, self.seed, self.witness, plan=self.plan, id_offset=LINEAGE_OFFSET, budget=second_budget
        )
        self.lines.append(second_line)
        second = await second_line.run_candidate()
        for index, line in enumerate(self.lines):
            self.trace.append({"lineage": index, "trace": line.trace})
        finals = [replace(w, name=f"L{index}:{w.name}") for index, w in enumerate((first, second)) if w is not None]
        if not finals:
            return fallback_module(self.photo)
        if len(finals) == 1:
            return finals[0].source
        best = (await self.judge.select(finals, 1, set(), FINAL_SELECT_ID))[0]
        self.trace.append(
            {"final": [c.name for c in finals], "winner": best.name, "select": (self.judge.trace or [{}])[-1]}
        )
        return best.source


# Submission profile used by main.py. Single-lineage plans only (sequential, replay-deterministic, <= 60 calls).
#   "funnel-lite"      F1.2.1-lite-prod: 6 Sol-high drafts, 5 all-Sol-high rounds (+39.2% on the 30-prompt benchmark)
#   "funnel"           F1-prod: original F1 plan (Astra round 2, xhigh rounds 3-5) on the hardened code (+36.7%)
#   "funnel-lite-deep" lite + 10 more Sol-high rounds (15 rounds, 57 calls; +36.7%)
PRODUCTION_PROFILE = "funnel-lite-deep"


async def generate(image: bytes, seed: int, witness) -> str:
    """Production entry (main.py): one funnel lineage with the plan named by PRODUCTION_PROFILE."""
    pipeline = FunnelPipeline(image, seed, witness, plan=PLANS[PRODUCTION_PROFILE])
    try:
        return await pipeline.run()
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError):
        # A bug in a late stage must not cost the prompt; infrastructure errors still propagate for recovery.
        LOG.exception("funnel stage failed; returning the best module so far")
        return pipeline.best_so_far or fallback_module(pipeline.photo)
