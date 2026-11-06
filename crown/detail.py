"""Experimental full-resolution feedback with successive revision populations.

Uses only the supplied photo, witnessed model responses and witnessed renders.
No benchmark sources, scores or development notes enter generation.
"""

import base64
import io
import json
import logging
import math

from PIL import Image

from . import media, prompts
from .beam import FOCUSES
from .budget import BudgetStop
from .experimental import EfficientPipeline
from .pipeline import Pipeline, json_object

LOG = logging.getLogger(__name__)

RANK = """Compare procedural 3D objects to the reference photograph. For each stable
candidate ID you receive FOUR separate 1024px white-background views in the named
camera order and ONE gray-background view. Judge the depicted object, ignoring
image framing, background, shadows and incidental illumination. Check silhouette,
proportions, pose, counted parts, openings, colors and visible material definition.
Use gray to check faint transparent or pale parts. A washed-out silhouette is a
material defect when the reference has defined edges, flutes or facets. Fine
detail does not compensate for a wrong outline. Do not prefer the incumbent or
new code merely because it is incumbent or new: candidate IDs are neutral.
Return JSON only: {"penalties":{"C0":number,...},"issues":"brief concrete defects
by candidate ID"}. Include EVERY supplied ID with a finite 0-10 likeness penalty:
0 faithful, 3 small discrepancies, 5 substantial errors, 7 wrong related object,
10 unrelated. Keep issues under 180 words."""


def vision_image(png: bytes, limit: int = 90 * 1024) -> dict:
    """Bound per-view payloads so a complete four-candidate decision fits 8 MiB."""
    if limit < 1024:
        raise ValueError("Per-view payload limit must be at least 1024 bytes")
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    while True:
        for quality in (94, 88, 80, 70, 58, 45):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, subsampling=0, optimize=False)
            if output.tell() <= limit:
                return {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(),
                    "detail": "high",
                }
        image = image.resize((max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)), Image.Resampling.LANCZOS)


class DetailPipeline(Pipeline):
    def ranking_instructions(self):
        return RANK

    async def detail_views(self, candidate):
        grid = await self.render(candidate, media.options(size=1024))
        white = media.panels(grid, size=1024)
        gray = await self.render(candidate, media.options(((30, 0),), size=1024, background="808080"))
        return white, gray

    async def rank(self, candidates):
        views = [await self.detail_views(candidate) for candidate in candidates]
        requests = []
        for order in (range(len(candidates)), reversed(range(len(candidates)))):
            content = [media.text("Reference photograph:"), media.data_image(self.photo)]
            for index in order:
                white, gray = views[index]
                for name, png in zip(media.ANGLE_NAMES, white):
                    content += [media.text(f"Candidate C{index}, white {name}:"), vision_image(png)]
                content += [media.text(f"Candidate C{index}, gray front-left:"), vision_image(gray)]
            requests.append((self.config.selector, self.ranking_instructions(), content, False, "medium"))
        try:
            replies = await EfficientPipeline.batch(self, requests)
        except BudgetStop:
            return None
        scores = []
        for reply in replies:
            result = json_object(reply)
            penalties = result.get("penalties") if result else None
            if not isinstance(penalties, dict):
                return None
            values = [penalties.get(f"C{i}") for i in range(len(candidates))]
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10 for value in values
            ):
                return None
            scores.append(values)
        return [(scores[0][i] + scores[1][i]) / 2 for i in range(len(candidates))]

    async def select(self, candidates):
        if len(candidates) == 1:
            return candidates[0]
        scores = await self.rank(candidates)
        if scores is None:
            return candidates[0]
        LOG.info("detail_ranking=%s", [(candidate.label, value) for candidate, value in zip(candidates, scores)])
        return candidates[min(range(len(candidates)), key=lambda i: (scores[i], i))]

    async def feedback(self, parent):
        white, gray = await self.detail_views(parent)
        sides = await self.render(parent, media.options(((90, 0), (180, 0), (270, 0), (0, -90)), size=512))
        content = [media.text("Reference photograph:"), media.data_image(self.photo)]
        for name, png in zip(media.ANGLE_NAMES, white):
            content += [media.text("1024px white " + name + ":"), vision_image(png)]
        content += [
            media.text("Gray front-left:"),
            vision_image(gray),
            media.text("Right / back / left / top, a four-view grid:"),
            vision_image(sides),
            media.text("Reference observations: " + self.plan),
        ]
        return content

    def revision_requests(self, common, parent, critique, step, models):
        requests = []
        for index, model in enumerate(models):
            content = [
                *common,
                media.text("Visual diagnosis (verify against the actual views): " + (critique or "None")),
                media.text(f"Revision wave {step + 1}, independent focus: " + FOCUSES[index]),
                media.text(prompts.REFINE + "\nCurrent source:\n" + parent.source),
            ]
            requests.append((model, prompts.ARTIST, content, True, "high" if model == "gpt-6-astra" else "xhigh"))
        return requests

    async def revisions(self, common, parent, critique, step):
        requests = self.revision_requests(common, parent, critique, step, ("gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-sol"))
        return await EfficientPipeline.batch(self, requests)

    async def run(self):
        # Keep initial requests identical to the original native control. Reusing
        # an exact response still charges its complete competition token usage.
        try:
            plan = json_object(await self.ask(self.config.planner, prompts.PLAN, [media.data_image(self.photo)]))
            if plan is not None:
                self.plan = json.dumps(plan, sort_keys=True, ensure_ascii=False)
        except BudgetStop:
            pass
        candidates = []
        for index, model in enumerate(self.config.drafts):
            candidate = await self.draft(model, index)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        parent = await self.select(candidates)
        for step in range(self.config.final_refinements):
            try:
                common = await self.feedback(parent)
                critique = await self.ask(self.config.planner, prompts.CRITIC, common, effort="medium")
                replies = await self.revisions(common, parent, critique, step)
            except BudgetStop:
                break
            pool = [parent]
            for index, reply in enumerate(replies):
                child, _ = await self.accept(reply, f"detail-{step}-{index}")
                if child is not None and all(child.source != old.source for old in pool):
                    pool.append(child)
            selected = await self.select(pool)
            if selected is not parent and await self.guard(parent, selected):
                LOG.info("detail_promoted=%s", selected.label)
                parent = selected
        return parent.source
