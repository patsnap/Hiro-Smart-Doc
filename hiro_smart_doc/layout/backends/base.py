from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Backend(Enum):
    ONNX = "onnx"


class RunnerBackend(ABC):
    @abstractmethod
    async def _inference(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]: ...
