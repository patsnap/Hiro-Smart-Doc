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


conf_thres = float(os.getenv("CONFIDENCE_THTRESHOLD_CHEM", 0.3))


class ModelRunnerChem(LayoutModelRunner):
    def __init__(self, backend: Backend, model: Path, threads: int) -> None:
        self.input_size = 960
        self.layout_classes =  [
            ("chem", (0, 139, 173)),
            ("rxn", (211, 219, 92))
        ]

        self.num_classes = 2
        self.complex_class = [0,1]
        self.conf_thres = conf_thres
        super().__init__(backend, model, threads, input_size=self.input_size, layout_classes=self.layout_classes, num_classes=self.num_classes, complex_class=self.complex_class, conf_thres=self.conf_thres)

        
