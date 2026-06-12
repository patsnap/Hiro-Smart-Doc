"""Standalone Gradio UI for Hiro-Smart-Doc.

Runs as its own process (NOT mounted on the gunicorn API workers) and calls the
in-process smart-doc pipeline directly. Exposes the user-facing parameter surface:
image / pdf / document inputs, layout/ocr/upload filters, PDF DPI, page
parallelism, markdown. (model_id and other defaults are fixed, not user-selectable.)

Launch:  python -m hiro_smart_doc.gradio_ui
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import cv2
import gradio as gr
import numpy as np

from .common.file_utils import load_image, render_pdf_page_at_index
from .service import (
    DETAILED_DEFAULT,
    Filter,
    REJECT_ASPECT_RATIO_DEFAULT,
    SIGN_URL_DEFAULT,
    _pdf_smart_doc_response,
    model_id,
    runner_dispatcher,
    smart_doc,
)

logger = logging.getLogger("gradio_ui")

LAYOUT_CATEGORIES = [
    "main_text",
    "table",
    "chemical",
    "equation",
    "figure",
    "supplemental_text",
    "complex",
    "others",
]


def _filter_from_selection(selected: list[str]) -> Filter:
    """Build a Filter where only the selected categories are True."""
    return Filter(**{cat: (cat in selected) for cat in LAYOUT_CATEGORIES})


def _draw_boxes(image: np.ndarray, textboxes: list[dict]) -> np.ndarray:
    """Draw normalized bboxes (TextBox dicts) onto a copy of the BGR image."""
    h, w = image.shape[:2]
    vis = image.copy()
    for tb in textboxes:
        bbox = tb.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (
            int(bbox[0] * w),
            int(bbox[1] * h),
            int(bbox[2] * w),
            int(bbox[3] * h),
        )
        label = tb.get("category") or tb.get("type") or ""
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 139, 173), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (13, 56, 212),
            1,
            cv2.LINE_AA,
        )
    return vis


def _parse_stream_lines(chunks: list[str]) -> tuple[list[dict], str]:
    """Split collected stream chunks into TextBox dicts and trailing markdown."""
    textboxes: list[dict] = []
    markdown = ""
    for raw in chunks:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "markdown" in obj:
                markdown = obj["markdown"]
            elif isinstance(obj, dict) and "bbox" in obj:
                textboxes.append(obj)
    return textboxes, markdown


async def _run_image(
    image_bgr: np.ndarray,
    layout_sel: list[str],
    ocr_sel: list[str],
    upload_sel: list[str],
    markdown: bool,
) -> tuple[list[str], str]:
    chunks: list[str] = []
    results = smart_doc(
        image_bgr,
        "gradio-image",
        0,
        _filter_from_selection(layout_sel).model_dump(),
        _filter_from_selection(ocr_sel).model_dump(),
        _filter_from_selection(upload_sel).model_dump(),
        model_id,
        DETAILED_DEFAULT,
        SIGN_URL_DEFAULT,
        "gradio",
        REJECT_ASPECT_RATIO_DEFAULT,
        markdown=markdown,
        last_page=True,
    )
    async for r in results:
        chunks.append(r)
    return chunks, ""


async def _run_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    layout_sel: list[str],
    ocr_sel: list[str],
    upload_sel: list[str],
    dpi: int,
    parallelism: int,
    markdown: bool,
) -> list[str]:
    response = await _pdf_smart_doc_response(
        pdf_bytes,
        filename,
        _filter_from_selection(layout_sel),
        _filter_from_selection(ocr_sel),
        _filter_from_selection(upload_sel),
        dpi,
        parallelism,
        upload_type="gradio",
        markdown=markdown,
    )
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return chunks


# ==================== Gradio handlers ====================


def handle_image(
    image, layout_sel, ocr_sel, upload_sel, markdown
):
    if image is None:
        return None, "请上传图片", "[]"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    chunks, _ = asyncio.run(
        _run_image(
            image_bgr, layout_sel, ocr_sel, upload_sel, markdown
        )
    )
    textboxes, md = _parse_stream_lines(chunks)
    vis = _draw_boxes(image_bgr, textboxes)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, (md or "(无 markdown 输出)"), json.dumps(
        textboxes, ensure_ascii=False, indent=2
    )


def handle_document(
    file_path,
    layout_sel,
    ocr_sel,
    upload_sel,
    dpi,
    parallelism,
    markdown,
):
    if not file_path:
        return None, "请上传 PDF / 文档", "[]"
    with open(file_path, "rb") as f:
        data = f.read()
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext != ".pdf":
        # Convert office documents to PDF first via the document pipeline.
        from .common.document_converter import (
            DocumentConversionError,
            convert_document_to_pdf,
        )

        try:
            converted = asyncio.run(convert_document_to_pdf(data, filename))
        except DocumentConversionError as e:
            return None, f"文档转换失败：{e}", "[]"
        data = converted.pdf_bytes
        filename = converted.pdf_filename

    chunks = asyncio.run(
        _run_pdf_bytes(
            data,
            filename,
            layout_sel,
            ocr_sel,
            upload_sel,
            int(dpi),
            int(parallelism),
            markdown,
        )
    )
    textboxes, md = _parse_stream_lines(chunks)

    # Visualize the first page for a quick preview.
    preview = None
    try:
        first = render_pdf_page_at_index(data, 0, int(dpi))
        page0 = [tb for tb in textboxes if tb.get("page") == 0]
        preview = cv2.cvtColor(_draw_boxes(first, page0), cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001
        logger.warning("preview render failed: %s", e)

    return preview, (md or "(无 markdown 输出)"), json.dumps(
        textboxes, ensure_ascii=False, indent=2
    )


# ==================== Gradio Blocks ====================


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hiro-Smart-Doc") as demo:
        gr.Markdown("# Hiro-Smart-Doc 交互体验\n版面分析 + OCR 智能文档解析")

        with gr.Row():
            with gr.Column(scale=1):
                layout_sel = gr.CheckboxGroup(
                    LAYOUT_CATEGORIES,
                    value=["main_text", "table", "equation"],
                    label="layout 过滤 (返回哪些类别)",
                )
                ocr_sel = gr.CheckboxGroup(
                    LAYOUT_CATEGORIES,
                    value=["main_text", "table", "equation"],
                    label="OCR 过滤 (对哪些类别做 OCR)",
                )
                upload_sel = gr.CheckboxGroup(
                    LAYOUT_CATEGORIES,
                    value=["figure"],
                    label="本地保存 (保存哪些类别裁切图)",
                )
                dpi = gr.Slider(72, 300, value=150, step=1, label="PDF DPI")
                parallelism = gr.Slider(
                    1, 8, value=8, step=1, label="PDF 页并行数"
                )
                markdown_cb = gr.Checkbox(value=True, label="生成 markdown")

            with gr.Column(scale=2):
                with gr.Tab("图片"):
                    img_in = gr.Image(type="numpy", label="上传图片")
                    img_btn = gr.Button("解析图片", variant="primary")
                    img_vis = gr.Image(label="版面可视化")
                with gr.Tab("PDF / 文档"):
                    doc_in = gr.File(
                        label="上传 PDF / Office 文档",
                        file_types=[
                            ".pdf", ".doc", ".docx", ".ppt", ".pptx",
                            ".xls", ".xlsx", ".odt", ".odp", ".ods", ".rtf",
                        ],
                    )
                    doc_btn = gr.Button("解析文档", variant="primary")
                    doc_vis = gr.Image(label="首页版面可视化")

                md_out = gr.Markdown(label="Markdown 输出")
                json_out = gr.Code(label="TextBox JSON", language="json")

        img_btn.click(
            handle_image,
            inputs=[img_in, layout_sel, ocr_sel, upload_sel, markdown_cb],
            outputs=[img_vis, md_out, json_out],
        )
        doc_btn.click(
            handle_document,
            inputs=[doc_in, layout_sel, ocr_sel, upload_sel,
                    dpi, parallelism, markdown_cb],
            outputs=[doc_vis, md_out, json_out],
        )

    return demo


def main() -> None:
    logging.basicConfig(level=os.getenv("RD_LOG_LEVEL", "INFO"))
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo = build_demo()
    demo.queue().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
