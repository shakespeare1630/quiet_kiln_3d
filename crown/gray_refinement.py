"""Experimental gray-view tie rescue for refinement decisions."""

from . import media
from .pipeline import Candidate, Pipeline, is_improvement


class GrayRefinementPipeline(Pipeline):
    refining = False

    async def refine(self, parent, step):
        self.refining = True
        try:
            return await super().refine(parent, step)
        finally:
            self.refining = False

    async def compare(self, first, second):
        white = await super().compare(first, second)
        if not self.refining or not white.complete or white.winner != 0 or is_improvement(white, self.config):
            return white
        # A full four-angle AB/BA comparison is still required. Keeping the same
        # thresholds avoids treating a convenient partial view as a quality win.
        gray = []
        for candidate in (first, second):
            grid = await self.render(candidate, media.options(background="808080"))
            gray.append(Candidate(candidate.source, grid, media.panels(grid), candidate.label + "-gray"))
        result = await super().compare(*gray)
        return result if result.complete and (result.winner != 0 or is_improvement(result, self.config)) else white
