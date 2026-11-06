"""Deterministic generation decisions over witnessed responses and renders.

Calls run in canonical order within a prompt. Parallelism belongs to the shared
batch runner, so scheduling, completion timing and wall clocks cannot choose a
candidate or alter a request. Infrastructure errors propagate for recovery.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field

from witness_client import RenderOptions, RenderRejected

from . import media, prompts
from .budget import Budget, BudgetStop, EvidenceUnavailable, reservation
from .config import DEFAULT, Config
from .source import extract, normalize

LOG = logging.getLogger(__name__)


def json_object(reply: str | None) -> dict | None:
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        return None
    try:
        value = json.loads(match[0], parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def penalties(reply: str | None) -> tuple[float, float] | None:
    data = json_object(reply)
    if data is None:
        return None
    values = [data.get("penalty_1"), data.get("penalty_2")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 10 for v in values):
        return None
    return float(values[0]), float(values[1])


@dataclass
class Candidate:
    source: str
    front: bytes
    views: tuple[bytes, ...]
    label: str
    renders: dict[RenderOptions, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    # -1: first candidate wins, +1: second wins, 0: draw or incomplete evidence.
    winner: int
    first_penalty: float
    second_penalty: float
    complete: bool
    consistent_angles: int = 0
    first_votes: int = 0
    second_votes: int = 0


def is_improvement(result: Comparison, config: Config) -> bool:
    unanimous_gain = (
        result.consistent_angles == result.second_votes == 4
        and result.first_penalty - result.second_penalty >= config.refinement_consensus_gap
        and (result.first_penalty + result.second_penalty) / 2 < 8
    )
    return result.complete and (result.winner == 1 or unanimous_gain)


def aggregate_comparisons(replies: list[tuple[float, float]], config: Config) -> Comparison:
    consistent = []
    for ab, ba in zip(replies[::2], replies[1::2]):
        first, second = (ab[0] + ba[1]) / 2, (ab[1] + ba[0]) / 2
        # Reverse BA back into the original candidate identity before checking
        # agreement. Contradictory presentation orders are not quality evidence.
        if (ab[0] - ab[1]) * (ba[1] - ba[0]) < 0:
            continue
        consistent.append((first, second))
    if len(consistent) < config.min_consistent_angles:
        return Comparison(0, 5, 5, True, len(consistent))
    first_penalty = sum(a for a, _ in consistent) / len(consistent)
    second_penalty = sum(b for _, b in consistent) / len(consistent)
    gap = second_penalty - first_penalty
    first_votes = sum(b > a for a, b in consistent)
    second_votes = sum(a > b for a, b in consistent)
    signed = first_votes + second_votes
    both_bad = (first_penalty + second_penalty) / 2 >= 8 and abs(gap) <= 1.5
    split = signed >= 3 and abs(first_votes - second_votes) / signed < 0.65
    winner = 0
    if not both_bad and not split:
        if gap > config.selection_gap:
            winner = -1
        elif gap < -config.selection_gap:
            winner = 1
    return Comparison(winner, first_penalty, second_penalty, True, len(consistent), first_votes, second_votes)


class Pipeline:
    def __init__(self, image: bytes, seed: int, witness, config: Config = DEFAULT):
        self.photo = media.canonical(image)
        self.seed = seed
        self.witness = witness
        self.config = config
        self.budget = Budget(config.max_credits, config.max_calls)
        self.request_id = 0
        self.plan = "Use direct visual evidence; no separate plan was available."

    async def ask(self, model: str, instructions: str, content: list[dict], *, code=False, effort="low") -> str | None:
        tokens = self.config.generation_reserve_tokens if code else self.config.decision_reserve_tokens
        expected = reservation(
            model,
            instructions,
            content,
            tokens,
            image_tokens=self.config.image_reserve_tokens,
            framing_tokens=self.config.framing_reserve_tokens,
        )
        self.budget.check(expected)
        self.request_id += 1
        request_id = self.request_id
        kwargs = dict(
            request_id=request_id,
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": effort},
            service_tier="default",
            store=False,
        )
        if self.config.send_output_limit:
            kwargs["max_output_tokens"] = tokens
        response = await self.witness.responses.create(**kwargs)
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            raise EvidenceUnavailable("The captured response has no verified usage")
        self.budget.settle(request_id, model, usage)
        LOG.info(
            "request=%d model=%s calls=%d credits=%s", request_id, model, len(self.budget.charges), self.budget.spent
        )
        if getattr(response, "status", None) != "completed":
            return None
        output = getattr(response, "output_text", None)
        return output if isinstance(output, str) else None

    async def render(self, candidate: Candidate, options: RenderOptions) -> bytes:
        if options not in candidate.renders:
            candidate.renders[options] = await self.witness.render(candidate.source, options)
        return candidate.renders[options]

    async def accept(self, reply: str | None, label: str) -> tuple[Candidate | None, str]:
        if not reply:
            return None, "No complete JavaScript was returned"
        try:
            source = normalize(extract(reply))
            front = await self.witness.render(source, media.options())
        except (ValueError, RenderRejected) as exc:
            return None, str(getattr(exc, "detail_json", exc))[:2500]
        views = media.panels(front)
        candidate = Candidate(source, front, views, label)
        candidate.renders[media.options()] = front
        return candidate, ""

    async def draft(self, model: str, index: int) -> Candidate | None:
        content = [
            media.text("Reference object:"),
            media.data_image(self.photo),
            media.text("Observed structure:\n" + self.plan),
            media.text(prompts.STRATEGIES[index] + "\nReturn the complete module."),
        ]
        try:
            reply = await self.ask(model, prompts.ARTIST, content, code=True, effort="high")
            for attempt in range(self.config.repairs + 1):
                candidate, error = await self.accept(reply, f"draft-{index}")
                if candidate is not None or attempt == self.config.repairs:
                    return candidate
                content = [
                    media.data_image(self.photo),
                    media.text(
                        "Fix this rejected module without changing its intended object.\n"
                        + error
                        + "\nPrevious response:\n"
                        + (reply or "")
                    ),
                ]
                reply = await self.ask(self.config.reviewer, prompts.ARTIST, content, code=True, effort="high")
        except BudgetStop:
            return None
        return None

    def compare_content(self, first: Candidate, second: Candidate, angle: int, swapped: bool) -> list[dict]:
        a, b = (second, first) if swapped else (first, second)
        return [
            media.text("Reference target:"),
            media.data_image(self.photo),
            media.text("Model 1 from " + media.ANGLE_NAMES[angle]),
            media.data_image(a.views[angle]),
            media.text("Model 2 from " + media.ANGLE_NAMES[angle]),
            media.data_image(b.views[angle]),
        ]

    async def compare(self, first: Candidate, second: Candidate) -> Comparison:
        if first.source == second.source:
            return Comparison(0, 0, 0, True)
        calls = [self.compare_content(first, second, angle, swapped) for angle in range(4) for swapped in (False, True)]
        estimated = sum(
            reservation(
                self.config.selector,
                prompts.JUDGE,
                content,
                self.config.decision_reserve_tokens,
                image_tokens=self.config.image_reserve_tokens,
                framing_tokens=self.config.framing_reserve_tokens,
            )
            for content in calls
        )
        try:
            # Start a complete AB/BA four-angle comparison only if its reservation
            # fits. This avoids choosing from a convenient partial angle subset.
            self.budget.check(estimated, calls=len(calls))
        except BudgetStop:
            return Comparison(0, 5, 5, False)
        replies = []
        try:
            for content in calls:
                replies.append(penalties(await self.ask(self.config.selector, prompts.JUDGE, content)))
        except BudgetStop:
            return Comparison(0, 5, 5, False)
        if any(reply is None for reply in replies):
            return Comparison(0, 5, 5, False)
        return aggregate_comparisons(replies, self.config)

    async def select(self, candidates: list[Candidate]) -> Candidate:
        if len(candidates) == 1:
            return candidates[0]
        scores = [0] * len(candidates)
        penalties_sum = [0.0] * len(candidates)
        appearances = [0] * len(candidates)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                result = await self.compare(candidates[i], candidates[j])
                if not result.complete:
                    # An incomplete tournament cannot fairly rank candidates
                    # with unequal evidence. Keep the deterministic fallback.
                    return candidates[0]
                scores[i] += 1 if result.winner == -1 else -1 if result.winner == 1 else 0
                scores[j] += 1 if result.winner == 1 else -1 if result.winner == -1 else 0
                penalties_sum[i] += result.first_penalty
                penalties_sum[j] += result.second_penalty
                appearances[i] += 1
                appearances[j] += 1
        # Stable source order breaks exact ties. Budget does not become a vote.
        best = min(
            range(len(candidates)),
            key=lambda i: (-scores[i], penalties_sum[i] / appearances[i] if appearances[i] else 5.0, i),
        )
        return candidates[best]

    async def full_views(self, candidate: Candidate) -> tuple[bytes, bytes]:
        sides = await self.render(candidate, media.options(((90, 0), (180, 0), (270, 0), (0, -90)), size=384))
        gray = await self.render(candidate, media.options(((30, 0),), size=384, background="808080"))
        return sides, gray

    async def guard(self, parent: Candidate, child: Candidate) -> bool:
        old_sides, old_gray = await self.full_views(parent)
        new_sides, new_gray = await self.full_views(child)
        content = [media.text("Reference:"), media.data_image(self.photo), media.text("Checklist: " + self.plan)]
        for label, candidate, sides, gray in (
            ("Candidate 1 (incumbent)", parent, old_sides, old_gray),
            ("Candidate 2 (proposed replacement)", child, new_sides, new_gray),
        ):
            content.extend(
                [
                    media.text(label + ": front / sides / gray"),
                    media.data_image(candidate.front),
                    media.data_image(sides),
                    media.data_image(gray),
                ]
            )
        try:
            result = json_object(await self.ask(self.config.planner, prompts.GUARD, content, effort="medium"))
        except BudgetStop:
            return False
        return (
            result is not None
            and result.get("candidate_2_regresses") is False
            and result.get("candidate_2_broken_3d") is False
        )

    async def refine(self, parent: Candidate, step: int) -> Candidate:
        try:
            sides, gray = await self.full_views(parent)
            critique = None
            if self.config.visual_critique:
                critique = json_object(
                    await self.ask(
                        self.config.planner,
                        prompts.CRITIC,
                        [
                            media.text("Reference:"),
                            media.data_image(self.photo),
                            media.text("Front views:"),
                            media.data_image(parent.front),
                            media.text("Right / back / left / top:"),
                            media.data_image(sides),
                            media.text("Gray front:"),
                            media.data_image(gray),
                            media.text("Observed checklist: " + self.plan),
                        ],
                        effort="medium",
                    )
                )
            content = [
                media.text("Reference:"),
                media.data_image(self.photo),
                media.text("Four front views:"),
                media.data_image(parent.front),
                media.text("Right / back / left / top:"),
                media.data_image(sides),
                media.text("Gray front:"),
                media.data_image(gray),
                media.text("Observed checklist: " + self.plan),
                media.text(
                    "Coarse saturated-pixel palette (not exact material colors):\nTarget: "
                    + media.color_summary(self.photo)
                    + "\nCurrent front views: "
                    + media.color_summary(parent.front)
                ),
                media.text(prompts.REFINE + "\nCurrent source:\n" + parent.source),
            ]
            if critique is not None:
                content.append(
                    media.text(
                        "Independent visual diagnosis (verify against the images):\n"
                        + json.dumps(critique, sort_keys=True, ensure_ascii=False)
                    )
                )
            model = self.config.final_models[step % len(self.config.final_models)]
            reply = await self.ask(model, prompts.ARTIST, content, code=True, effort="high")
            child, _ = await self.accept(reply, f"refine-{step}")
            if child is None or child.source == parent.source:
                return parent
            result = await self.compare(parent, child)
            if is_improvement(result, self.config) and await self.guard(parent, child):
                LOG.info("promoted=%s", child.label)
                return child
        except (BudgetStop, RenderRejected):
            pass
        return parent

    async def run(self) -> str | None:
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
        winner = await self.select(candidates)
        for step in range(self.config.final_refinements):
            winner = await self.refine(winner, step)
        return winner.source


async def generate(image: bytes, seed: int, witness) -> str | None:
    return await Pipeline(image, seed, witness).run()
