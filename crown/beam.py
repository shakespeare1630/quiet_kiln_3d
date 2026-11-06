"""Experimental parallel high-reasoning revision siblings with grouped selection."""

import json

from . import media, prompts
from .astra_selection import AstraSelectionPipeline
from .budget import BudgetStop
from .experimental import EfficientPipeline
from .pipeline import json_object

FOCUSES = (
    "Correct the largest measured geometric mismatch: pose, main proportions, outline, openings or counted parts. Preserve accurate materials.",
    "Correct visible surface/material definition in the actual neutral renders, including overly faint clear parts or dull facets. Preserve accurate geometry and counts.",
    "Produce an integrated correction of the two most important concrete visual defects. Use precise profiles and real thickness; preserve every already-correct distinctive feature.",
)


class BeamPipeline(AstraSelectionPipeline):
    async def run(self):
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
        sides, gray = await self.full_views(parent)
        common = [
            media.text("Reference:"),
            media.data_image(self.photo),
            media.text("Four front views:"),
            media.data_image(parent.front),
            media.text("Right / back / left / top:"),
            media.data_image(sides),
            media.text("Gray front:"),
            media.data_image(gray),
            media.text("Observed checklist: " + self.plan),
        ]
        try:
            critique = await self.ask(self.config.planner, prompts.CRITIC, common, effort="medium")
            requests = []
            for index, focus in enumerate(FOCUSES):
                content = [
                    *common,
                    media.text("Independent diagnosis (verify visually): " + (critique or "No diagnosis")),
                    media.text(f"Independent revision {index + 1}: " + focus),
                    media.text(prompts.REFINE + "\nCurrent source:\n" + parent.source),
                ]
                requests.append(("gpt-6.1-sol", prompts.ARTIST, content, True, "xhigh"))
            replies = await EfficientPipeline.batch(self, requests)
        except BudgetStop:
            return parent.source
        pool = [parent]
        for index, reply in enumerate(replies):
            child, _ = await self.accept(reply, f"beam-{index}")
            if child is not None and all(child.source != old.source for old in pool):
                pool.append(child)
        if len(pool) == 1:
            return parent.source
        selected = await self.select(pool)
        if selected is not parent and await self.guard(parent, selected):
            return selected.source
        return parent.source
