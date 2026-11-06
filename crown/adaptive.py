"""Native detail experiment with budgeted Sol fallback and optical alternatives."""

import logging

from .budget import BudgetStop, reservation
from .detail import DetailPipeline
from .experimental import EfficientPipeline
from .optics import finish_optics

LOG = logging.getLogger(__name__)


class AdaptiveDetailPipeline(DetailPipeline):
    async def revisions(self, common, parent, critique, step):
        groups = [("gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-sol"), *(("gpt-6.1-sol",) * n for n in (3, 2, 1))]
        for models in groups:
            requests = self.revision_requests(common, parent, critique, step, models)
            expected = sum(
                reservation(
                    model,
                    instructions,
                    content,
                    self.config.generation_reserve_tokens,
                    image_tokens=self.config.image_reserve_tokens,
                    framing_tokens=self.config.framing_reserve_tokens,
                )
                for model, instructions, content, code, effort in requests
            )
            try:
                self.budget.check(expected + getattr(self, "finishing_reserve_credits", 0), calls=len(requests))
            except BudgetStop:
                continue
            LOG.info("revision_models=%s", models)
            # Infrastructure errors after admission propagate. The fallback is
            # chosen before making calls, never from a partially successful pool.
            return await EfficientPipeline.batch(self, requests)
        raise BudgetStop

    async def finish(self, parent):
        return await finish_optics(self, parent)

    async def run(self):
        source = await super().run()
        if source is None:
            return None
        parent, error = await self.accept(source, "reconstruction")
        if parent is None:
            raise RuntimeError(error)
        selected = await self.finish(parent)
        if selected is not parent:
            LOG.info("optics_promoted=%s", selected.label)
        return selected.source
