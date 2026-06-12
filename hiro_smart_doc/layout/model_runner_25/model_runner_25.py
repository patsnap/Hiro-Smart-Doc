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


conf_thres = float(os.getenv("CONFIDENCE_THTRESHOLD_25", 0.3))


class ModelRunner25(LayoutModelRunner):
    def __init__(self, backend: Backend, model: Path, threads: int) -> None:
        self.input_size = 640
        self.layout_classes =  [("title", (97, 242, 211)),
            ("sec", (13, 158, 56)),
            ("text", (0, 139, 173)),
            ("photo", (13, 158, 56)),
            ("seq", (0, 139, 173)),
            ("head", (211, 219, 92)),
            ("foot", (217, 109, 9)),
            ("draw", (13, 56, 212)),
            ("mnote", (222, 84, 146)),
            ("cap", (171, 89, 247)),
            ("struc", (0, 139, 173)),
            ("figno", (158, 163, 255)),
            ("lineno", (13, 56, 212)),
            ("colno", (105, 192, 255)),
            ("ref", (13, 56, 212)),
            ("toc", (158, 163, 255)),
            ("noise", (105, 192, 255)),
            ("tab", (211, 219, 92)),
            ("eqn", (217, 109, 9)),
            ("chem", (255, 198, 173)),
            ("figcx", (97, 242, 211)),
            ("rxn", (13, 158, 56)),
            ("bib", (211, 219, 92)),
            ("srep", (217, 109, 9)),
            ("graph", (158, 163, 255)),
        ]

        self.num_classes = 25
        self.complex_class = [11, 19, 20, 21, 22, 23]
        self.conf_thres = conf_thres
        super().__init__(backend, model, threads, input_size=self.input_size, layout_classes=self.layout_classes, num_classes=self.num_classes, complex_class=self.complex_class, conf_thres=self.conf_thres)

        
