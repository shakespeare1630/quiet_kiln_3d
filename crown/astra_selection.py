"""Experimental grouped Astra decisions with white and gray view evidence."""

import math
from dataclasses import replace

from . import media
from .budget import BudgetStop
from .experimental import EfficientPipeline
from .pipeline import Comparison, Pipeline, json_object

ASTRA_RANK = """Assess procedural 3D reconstructions against the reference photo.
Each candidate ID has four white-background views and four gray-background views.
Judge the photographed object: pose, outline, proportions, counted components,
openings, color, gloss, transparency and surface definition. Use both backgrounds
to see pale/clear parts. Ignore background differences and incidental lighting.
Prefer an overall closer reconstruction even if a small individual detail differs.
Do not reward a larger model, artistic style or additional unrelated details.
Return JSON only: a `penalties` object mapping EVERY candidate ID to a number from
0 (faithful) to 10 (unrelated), and an `issues` string explaining the main defects.
3 means minor deviations, 5 substantial errors, 7 wrong related object. Keep the
explanation under 150 words. IDs remain attached to their images in every order."""


class AstraSelectionPipeline(Pipeline):
    async def ranked(self, candidates):
        gray = [await self.render(candidate, media.options(background="808080")) for candidate in candidates]
        requests = []
        for order in (range(len(candidates)), reversed(range(len(candidates)))):
            content = [media.text("Reference:"), media.data_image(self.photo)]
            for index in order:
                content.extend(
                    [
                        media.text(f"Candidate C{index}, four white views:"),
                        media.data_image(candidates[index].front),
                        media.text(f"Candidate C{index}, four gray views:"),
                        media.data_image(gray[index]),
                    ]
                )
            requests.append((self.config.selector, ASTRA_RANK, content, False, "medium"))
        try:
            replies = await EfficientPipeline.batch(self, requests)
        except BudgetStop:
            return None
        scores = []
        for reply in replies:
            data = json_object(reply)
            values = data.get("penalties", {}) if data else {}
            if not isinstance(values, dict):
                return None
            if any(
                type(values.get(f"C{i}")) not in (int, float)
                or not math.isfinite(values[f"C{i}"])
                or not 0 <= values[f"C{i}"] <= 10
                for i in range(len(candidates))
            ):
                return None
            scores.append([values[f"C{i}"] for i in range(len(candidates))])
        return scores

    async def select(self, candidates):
        if len(candidates) == 1:
            return candidates[0]
        scores = await self.ranked(candidates)
        if scores is None:
            return candidates[0]
        winners = [min(range(len(candidates)), key=lambda i: (row[i], i)) for row in scores]
        return candidates[winners[0]] if winners[0] == winners[1] else candidates[0]

    async def compare(self, first, second):
        if first.source == second.source:
            return Comparison(0, 0, 0, True)
        scores = await self.ranked([first, second])
        if scores is None:
            return Comparison(0, 5, 5, False)
        first_penalty, second_penalty = ((scores[0][i] + scores[1][i]) / 2 for i in range(2))
        differences = [row[0] - row[1] for row in scores]
        winner = 0
        both_bad = (first_penalty + second_penalty) / 2 >= 8 and abs(first_penalty - second_penalty) <= 1.5
        if not both_bad and all(d > 0 for d in differences) and first_penalty - second_penalty >= 0.3:
            winner = 1
        elif not both_bad and all(d < 0 for d in differences) and second_penalty - first_penalty >= 0.3:
            winner = -1
        # These are two complete grouped judgments, not individual-angle votes.
        return Comparison(winner, first_penalty, second_penalty, True)

    async def guard(self, parent, child):
        original = self.config
        self.config = replace(original, planner=original.selector)
        try:
            return await super().guard(parent, child)
        finally:
            self.config = original
