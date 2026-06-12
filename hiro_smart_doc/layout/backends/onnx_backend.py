import asyncio
import logging
from pathlib import Path

import numpy as np
import onnxruntime  # type: ignore
from numpy.typing import NDArray

from .base import RunnerBackend

_dlls_preloaded = False


def _preload_cuda_dlls() -> None:
    """Load CUDA/cuDNN shared libs shipped as pip wheels (onnxruntime-gpu[cuda,cudnn]).

    Without this, onnxruntime cannot find libcudnn.so.9 unless it is installed
    system-wide. It is a no-op on CPU-only installs (no preload_dlls / no wheels).
    """
    global _dlls_preloaded
    if _dlls_preloaded:
        return
    _dlls_preloaded = True
    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as exc:  # pragma: no cover - best-effort, falls back to CPU
        logging.getLogger("onnx_backend").debug("preload_dlls skipped: %s", exc)


class OnnxBackend(RunnerBackend):
    def __init__(self, model: Path, intra_op_num_threads: int) -> None:
        self.logger = logging.getLogger("onnx_backend")
        self.logger.info(f"Loading {model=}, {intra_op_num_threads=}")

        _preload_cuda_dlls()

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads

        self.ort_session = onnxruntime.InferenceSession(
            model, sess_options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    async def _inference(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        inputs = {self.input_name: tensor}
        return (await asyncio.to_thread(self.ort_session.run, None, inputs))[0]  # type: ignore
