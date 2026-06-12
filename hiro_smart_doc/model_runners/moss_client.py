"""
Lightweight OpenAI-compatible client for the MOSS-OCR model.

Hiro-Smart-Doc only acts as an HTTP client: it talks to a separately deployed
vLLM server that serves the MOSS-OCR model. See the Hiro-MOSS-OCR project for
how to serve the model: https://github.com/patsnap/Hiro-MOSS-OCR

This module intentionally keeps no model/training code — just enough to send
images to an OpenAI-compatible /v1/chat/completions endpoint and collect text.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

_logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    result: Any
    task: str
    is_succeed: bool
    time_cost: float
    model_name: str
    is_repeated: bool | None = None
    error_message: str | None = None


def _to_numpy(img: np.ndarray | str) -> np.ndarray:
    """Accept a BGR ndarray (as produced by cv2.imread) or a file path."""
    if isinstance(img, np.ndarray):
        return img
    arr = cv2.imread(img)
    if arr is None:
        raise ValueError(f"cannot read image: {img}")
    return arr


def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _truncate_repetitions(text: str, min_len: int = 15) -> str:
    """Trim a repeating tail from model output (adapted from nougat)."""
    if len(text) < 2 * min_len:
        return text
    max_rep_len = None
    for rep_len in range(min_len, int(len(text) / 2)):
        same = True
        for i in range(0, rep_len):
            if text[len(text) - rep_len - i - 1] != text[len(text) - i - 1]:
                same = False
                break
        if same:
            max_rep_len = rep_len
    if max_rep_len is None:
        return text
    lcs = text[-max_rep_len:]
    truncated = text
    while truncated.endswith(lcs):
        truncated = truncated[:-max_rep_len]
    return text[: len(truncated)]


class MOSSOCRPipeline:
    """Async OpenAI-compatible client for a vLLM-served MOSS-OCR model."""

    TASK_PROMPT_MAP = {
        "math": "read formula from image and output in Latex formula format: \n",
        "table": "read table from image and output in HTML format: \n",
        "text": "read text from image and output in Markdown format: \n",
    }
    MODEL_NAME = "moss-v1d6-0.3b"

    def __init__(
        self,
        model_path: str | None = None,
        max_length: int = 2048,
        max_retry: int = 3,
        max_concurrent: int = 32,
        url: str = "http://127.0.0.1:8088/v1",
        api_key: str = "EMPTY",
        timeout: int = 3600,
        detect_repeat: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_name = model_path or self.MODEL_NAME
        self.max_length = max_length
        self.max_retry = max_retry
        self.max_concurrent = max_concurrent
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.detect_repeat = detect_repeat
        self.logger = logger or _logger

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: AsyncOpenAI | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """One semaphore per running loop; caps in-flight OCR requests."""
        loop = asyncio.get_running_loop()
        if self._sem_loop is not loop:
            self._sem_loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        assert self._semaphore is not None
        return self._semaphore

    def _get_client(self) -> AsyncOpenAI:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            limits = httpx.Limits(
                max_connections=self.max_concurrent + 10,
                max_keepalive_connections=self.max_concurrent,
            )
            http_client = httpx.AsyncClient(limits=limits, timeout=self.timeout)
            self._client = AsyncOpenAI(
                base_url=self.url,
                api_key=self.api_key,
                http_client=http_client,
                max_retries=0,
            )
        assert self._client is not None
        return self._client

    def build_payload(
        self, img: np.ndarray | str, task: str, max_length: int | None = None
    ) -> dict:
        img_b64 = _ndarray_to_base64_png(_to_numpy(img))
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": self.TASK_PROMPT_MAP[task]},
                    ],
                }
            ],
            "max_completion_tokens": max_length or self.max_length,
            "temperature": 0,
            "top_p": 1.0,
        }

    async def async_run(
        self, img: np.ndarray | str, task: str, max_length: int | None = None
    ) -> OCRResult:
        max_length = max_length or self.max_length
        client = self._get_client()
        started = time.perf_counter()
        try:
            for attempt in range(self.max_retry):
                try:
                    payload = await asyncio.to_thread(
                        self.build_payload, img, task, max_length
                    )
                    async with self._get_semaphore():
                        response = await client.chat.completions.create(**payload)
                    result = response.choices[0].message.content or ""
                    is_repeated = None
                    if self.detect_repeat:
                        truncated = await asyncio.to_thread(
                            _truncate_repetitions, result
                        )
                        is_repeated = truncated != result
                        result = truncated
                    return OCRResult(
                        result=result,
                        task=task,
                        is_succeed=True,
                        time_cost=time.perf_counter() - started,
                        model_name=self.model_name,
                        is_repeated=is_repeated,
                    )
                except RateLimitError as e:
                    sleep = min(2**attempt, 60) + random.uniform(0, 1)
                    self.logger.warning(
                        "[%d/%d] rate limited, retrying in %.1fs: %s",
                        attempt + 1, self.max_retry, sleep, e,
                    )
                    await asyncio.sleep(sleep)
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    self.logger.warning(
                        "[%d/%d] network/server error, retrying: %s",
                        attempt + 1, self.max_retry, e,
                    )
                    await asyncio.sleep(random.uniform(1, 3))
                except BadRequestError as e:
                    if "Already borrowed" in str(e):
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                        continue
                    self.logger.error("bad request, not retrying: %s", e)
                    raise
                except (AuthenticationError, APIStatusError) as e:
                    self.logger.error("non-retryable API error: %s", e)
                    raise
            raise RuntimeError(f"OCR failed after {self.max_retry} retries")
        except Exception as e:
            self.logger.error(
                "OCR request failed: %s\n%s", e, traceback.format_exc()
            )
            return OCRResult(
                result=None,
                task=task,
                is_succeed=False,
                time_cost=time.perf_counter() - started,
                model_name=self.model_name,
                error_message=str(e),
            )

    async def async_run_batch(
        self,
        img_ls: list[np.ndarray | str],
        task_ls: list[str],
        max_length_ls: list[int] | None = None,
    ) -> list[OCRResult]:
        """Run OCR over a batch; concurrency is capped by the shared semaphore."""
        if max_length_ls is None:
            max_length_ls = [self.max_length] * len(img_ls)
        tasks = [
            self.async_run(img, task, max_length=length)
            for img, task, length in zip(img_ls, task_ls, max_length_ls)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)
