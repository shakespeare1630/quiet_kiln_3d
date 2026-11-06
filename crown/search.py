"""Experimental wider search; not the submission entry point.

The source, seed, witness and configuration fully determine request order. Draft
batches are bounded to three concurrent calls; no completion timing selects code.
"""

from witness_client import RenderRejected

from . import media
from .experimental import EfficientPipeline
from .fit import fit
from .pipeline import Candidate
from .source import extract


class WidePipeline(EfficientPipeline):
    async def batch(self, requests):
        if len(requests) <= 3:
            return await super().batch(requests)
        # Reserve the whole stage before dispatch, including subsequent batches.
        # EfficientPipeline's precheck also applies to each bounded batch.
        from .budget import reservation

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
        replies = []
        for start in range(0, len(requests), 3):
            replies.extend(await super().batch(requests[start : start + 3]))
        return replies

    async def accept(self, reply, label):
        if not reply:
            return None, "No complete JavaScript was returned"
        try:
            source = fit(extract(reply))
            front = await self.witness.render(source, media.options())
        except (ValueError, RenderRejected) as exc:
            return None, str(getattr(exc, "detail_json", exc))[:2500]
        candidate = Candidate(source, front, media.panels(front), label)
        candidate.renders[media.options()] = front
        return candidate, ""
