"""
MOSS OCR runner: HTTP client to a vLLM-served MOSS-OCR model.

The model itself is served separately (see https://github.com/patsnap/Hiro-MOSS-OCR).
This runner only sends images to that OpenAI-compatible endpoint via `moss_client`.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import cv2
from cv2.typing import MatLike

from hiro_smart_doc.common.stage_timing import StageRecorder

from .moss_client import MOSSOCRPipeline, OCRResult

# Layout categories -> MOSSOCRPipeline TASK_PROMPT_MAP keys (math | table | text)
CATEGORY_TO_TASK: dict[str, str] = {
    "table": "table",
    "equation": "math",
    "main_text": "text",
    "supplemental_text": "text",
}


def _ocr_result_to_text(ocr: OCRResult) -> str:
    if not ocr.is_succeed or ocr.result is None:
        return ""
    if isinstance(ocr.result, str):
        return ocr.result
    return str(ocr.result)


class MossOcrRunner:
    """OpenAI-compatible MOSS OCR (VLLM)."""

    def __init__(self) -> None:
        url = os.getenv("MOSS_VLLM_OCR_API", "http://127.0.0.1:8000/v1")
        api_key= os.getenv("MOSS_VLLM_OCR_API_KEY", "EMPTY")
        max_concurent= int(os.getenv("MOSS_VLLM_OCR_MAX_CONCURRENT", "32"))      
        model_path = os.getenv("MOSS_VLLM_MODEL", MOSSOCRPipeline.MODEL_NAME)
        self.moss_client = MOSSOCRPipeline(model_path=model_path, url=url, api_key=api_key, max_concurrent=max_concurent)
        self.logger = logging.getLogger("moss_ocr_runner")

    async def inference_batch(
        self,
        entries: list[tuple[MatLike, str]],
        *,
        page: int | None = None,
        pdf_path: str | None = None,
    ) -> list[str]:
        """
        Run batch OCR for table / math / text.
        entries: list of (image, category) with category in
        ("table", "equation", "main_text", "supplemental_text").
        Returns list of result strings in same order.
        """
        meta: dict[str, Any] = {}
        if page is not None:
            meta["page"] = page
        if pdf_path is not None:
            meta["pdf_path"] = pdf_path
        rec = StageRecorder("moss_vllm_batch", **meta)
        rec.mark("start")
        if not entries:
            return []
        rec.mark("prepare")
        img_ls: list[MatLike] = []
        task_ls: list[str] = []
        for img, category in entries:
            task = CATEGORY_TO_TASK.get(category, "text")
            if task not in self.moss_client.TASK_PROMPT_MAP:
                task = "text"
            img_ls.append(img)
            task_ls.append(task)
        max_length_ls = [self.moss_client.max_length] * len(img_ls)
        rec.mark("request")
        ocr_results = await self.moss_client.async_run_batch(
            img_ls, task_ls, max_length_ls
        )
        rec.mark("response")
        out = [_ocr_result_to_text(r) for r in ocr_results]
        for r in ocr_results:
            if not r.is_succeed:
                self.logger.error(
                    "MOSS VLLM OCR failed for task=%s: %s",
                    r.task,
                    r.error_message or "unknown",
                )
        rec.log_info(self.logger, payload=rec.finish())
        return out

    async def inference(self, image: MatLike, category: str) -> str:
        """Single-image inference; delegates to inference_batch."""
        results = await self.inference_batch([(image, category)])
        return results[0] if results else ""


async def _cli_main() -> None:
    runner = MossOcrRunner()
    example_dir = os.getenv("MOSS_OCR_EXAMPLE_DIR", "./example/images")
    img_paths = sorted(
        os.path.join(example_dir, f)
        for f in os.listdir(example_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    )
    entries: list[tuple[MatLike, str]] = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        entries.append((img, "main_text"))
    result = await runner.inference_batch(entries)
    for i in result:
        print(i)
        print("--------------------------------\n\n")


if __name__ == "__main__":
    asyncio.run(_cli_main())
