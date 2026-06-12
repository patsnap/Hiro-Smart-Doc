import logging
from functools import partial

from ..common.utils import retry
from .layout import LayoutRunner

from .moss_ocr_runner import MossOcrRunner

__all__ = [
    "LayoutRunner",
    "MossOcrRunner",
]


class RunnerDispatcher:
    def __init__(self) -> None:
        self.logger = logging.getLogger("runner_dispatcher")
        self.retry = partial(retry, 3, self.logger)

        self.layout_runner = LayoutRunner()
        self.moss_ocr_runner = MossOcrRunner()
        self.filter = self.layout_runner.filter

        # APIs
        self.layout_inference = self.retry("layout")(self.layout_runner.inference)
        self.moss_ocr_batch = self.retry("moss_ocr_batch")(
            self.moss_ocr_runner.inference_batch
        )

    def release(self) -> None:
        del self.moss_ocr_runner
        del self.layout_runner
