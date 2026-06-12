import logging
import math
import os
from pathlib import Path
import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from .. import infer_utils

from ..backends.base import Backend
from ..model_runner import LayoutModelRunner


conf_thres = float(os.getenv("CONFIDENCE_THTRESHOLD_9", 0.25))


class ModelRunner9(LayoutModelRunner):
    def __init__(self, backend: Backend, model: Path, threads: int) -> None:
        self.input_size = 960
        self.layout_classes =  [
                ("text", (0, 139, 173)),
                ("supplement", (13, 158, 56)),
                ("noise", (0, 139, 173)),
                ("tab", (211, 219, 92)),
                ("graph", (217, 109, 9)),
                ("fig", (13, 56, 212)),
                ("eqn", (217, 109, 9)),
                ("chem", (255, 198, 173)),
                ("rxn", (13, 158, 56)),
        ]

        self.num_classes = 9
        self.complex_class = [4, 5, 7, 8]
        self.conf_thres = conf_thres
        super().__init__(backend, model, threads, input_size=self.input_size, layout_classes=self.layout_classes, num_classes=self.num_classes, complex_class=self.complex_class, conf_thres=self.conf_thres)

        
