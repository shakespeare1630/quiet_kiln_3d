"""Experimental faster draft selection and measured render feedback.

Not the active submission entry point. Every decision still uses only the prompt,
fixed instructions, supplied witness responses and rendered generated geometry.
"""

import asyncio
import io
import json
import math
from statistics import median

from PIL import Image

from . import media, prompts
from .budget import BudgetStop, reservation
from .pipeline import Pipeline, json_object


def visible_summary(png):
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image.thumbnail((128, 128), Image.Resampling.BILINEAR)
    width, height = image.size
    pixels = list(image.get_flattened_data())
    border = pixels[:width] + pixels[-width:] + pixels[::width] + pixels[width - 1 :: width]
    background = tuple(int(median([p[i] for p in border])) for i in range(3))
    selected = [(i, p) for i, p in enumerate(pixels) if sum(abs(p[c] - background[c]) for c in range(3)) >= 30]
    if not selected:
        return {"background_rgb": background, "separated_fraction": 0, "note": "No reliable separated foreground"}
    xs = [i % width for i, _ in selected]
    ys = [i // width for i, _ in selected]
    return {
        "background_rgb": background,
        "separated_fraction": round(len(selected) / len(pixels), 3),
        "separated_median_rgb": [int(median([p[c] for _, p in selected])) for c in range(3)],
        "mean_distance_from_background": sum(sum(abs(p[c] - background[c]) for c in range(3)) for _, p in selected)
        // (3 * len(selected)),
        "outline_width_over_height": round((max(xs) - min(xs) + 1) / (max(ys) - min(ys) + 1), 3),
    }


RANK = """Compare the complete 3D reconstructions with the target. Candidate IDs
remain attached to their images. Each candidate image contains four camera views.
Judge likeness of pose, silhouette, proportions, counted components, openings,
and visible material/color contrast. Nearly invisible pale or transparent objects
are not accurate when the target has visible rims, reflections or shading.
Give a 0-10 penalty per candidate: 0 faithful, 3 minor defects, 5 substantial
errors, 7 wrong related object, 10 unrelated. Return JSON only, with a `penalties`
object mapping EVERY supplied candidate ID to its numeric penalty and an `issues`
string with brief differences. Use every supplied ID exactly; do not rate
aesthetic attractiveness."""


class EfficientPipeline(Pipeline):
    async def batch(self, requests):
        self.budget.check(
            sum(
                reservation(
                    model,
                    instructions,
                    content,
                    self.config.generation_reserve_tokens if code else self.config.decision_reserve_tokens,
                    image_tokens=self.config.image_reserve_tokens,
                    framing_tokens=self.config.framing_reserve_tokens,
                )
                for model, instructions, content, code, effort in requests
            ),
            calls=len(requests),
        )
        values = await asyncio.gather(
            *(
                self.ask(model, instructions, content, code=code, effort=effort)
                for model, instructions, content, code, effort in requests
            ),
            return_exceptions=True,
        )
        for value in values:
            if isinstance(value, BaseException):
                raise value
        return values

    async def select(self, candidates):
        if len(candidates) <= 2:
            return await super().select(candidates)
        requests = []
        for order in (list(range(len(candidates))), list(reversed(range(len(candidates))))):
            content = [media.text("Reference target:"), media.data_image(self.photo)]
            for i in order:
                content.extend([media.text(f"Candidate C{i}, four views:"), media.data_image(candidates[i].front)])
            requests.append((self.config.selector, RANK, content, False, "low"))
        try:
            replies = await self.batch(requests)
        except BudgetStop:
            return candidates[0]
        scores = []
        for reply in replies:
            data = json_object(reply)
            values = data.get("penalties", {}) if data else {}
            if not isinstance(values, dict):
                return await super().select(candidates)
            if any(
                type(values.get(f"C{i}")) not in (int, float)
                or not math.isfinite(values[f"C{i}"])
                or not 0 <= values[f"C{i}"] <= 10
                for i in range(len(candidates))
            ):
                return await super().select(candidates)
            scores.append([values[f"C{i}"] for i in range(len(candidates))])
        order = sorted(range(len(candidates)), key=lambda i: (scores[0][i] + scores[1][i], i))
        winners = [min(range(len(candidates)), key=lambda i: (row[i], i)) for row in scores]
        if winners[0] == winners[1]:
            return candidates[order[0]]
        first, second = (candidates[i] for i in order[:2])
        comparison = await self.compare(first, second)
        return second if comparison.complete and comparison.winner == 1 else first

    async def refine(self, parent, step):
        previous = self.plan
        self.plan = json.dumps(
            {
                "observations": json_object(previous) or previous,
                "coarse_reference_pixels": visible_summary(self.photo),
                "coarse_render_pixels": [visible_summary(view) for view in parent.views],
                "measurement_caution": "Background-distance segmentation includes shadows and may miss pale surfaces. Verify visually.",
                "material_correction": "Match visible contrast under these renders. Correct overly faint glass, missing rim/reflection definition, and overly dark metal using plausible material settings and real geometry, not flat artwork.",
            },
            sort_keys=True,
        )
        try:
            return await super().refine(parent, step)
        finally:
            self.plan = previous

    async def run(self):
        try:
            plan = json_object(await self.ask(self.config.planner, prompts.PLAN, [media.data_image(self.photo)]))
            if plan is not None:
                self.plan = json.dumps(plan, sort_keys=True, ensure_ascii=False)
        except BudgetStop:
            pass
        requests = []
        for i, model in enumerate(self.config.drafts):
            content = [
                media.text("Reference object:"),
                media.data_image(self.photo),
                media.text("Observed structure:\n" + self.plan),
                media.text(f"Independent draft {i + 1}: " + prompts.STRATEGIES[i] + " Return the complete module."),
            ]
            requests.append((model, prompts.ARTIST, content, True, "high"))
        try:
            replies = await self.batch(requests)
        except BudgetStop:
            return None
        candidates = []
        for i, reply in enumerate(replies):
            candidate, error = await self.accept(reply, f"draft-{i}")
            if candidate is None and self.config.repairs:
                repaired = await self.ask(
                    self.config.reviewer,
                    prompts.ARTIST,
                    [
                        media.data_image(self.photo),
                        media.text("Fix the rejected module.\n" + error + "\n" + (reply or "")),
                    ],
                    code=True,
                    effort="high",
                )
                candidate, _ = await self.accept(repaired, f"repair-{i}")
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        winner = await self.select(candidates)
        for step in range(self.config.final_refinements):
            winner = await self.refine(winner, step)
        return winner.source
