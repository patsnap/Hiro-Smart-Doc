import asyncio
import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator

from annotated_types import Len
from cv2.typing import MatLike
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from typing_extensions import Annotated

from .common.aiohttp_session import SingletonAiohttp
from .common.document_converter import (
    DocumentConversionDisabledError,
    DocumentConversionError,
    UnsupportedDocumentError,
    convert_document_to_pdf,
)
from .common.file_utils import (
    crop_image_bbox,
    get_aspect_ratio,
    hash_file,
    load_image,
    pdf_page_count,
    render_pdf_page_at_index,
    save_image,
)
from .common.local_storage import image_url, save_image_local
from .common.stage_timing import StageRecorder
from .common.utils import (
    CoroutinePool,
    aenumerate,
    content_disposition_attachment,
    log_error,
    now_time_str,
)
from .model_runners import RunnerDispatcher

logger = logging.getLogger(__name__)
runner_dispatcher = RunnerDispatcher()
req_parallelism = int(os.getenv("REQ_PARALLELISM", 2))
moss_chunk_parallelism = int(
    os.getenv("MOSS_CHUNK_PARALLELISM", str(req_parallelism))
)
pdf_page_parallelism_default = int(os.getenv("PDF_PAGE_PARALLELISM_DEFAULT", "8"))
pdf_page_parallelism_max = int(os.getenv("PDF_PAGE_PARALLELISM_MAX", "8"))
upload_prefix = os.getenv("UPLOAD_PREFIX", "")
aspect_ratio_range = tuple(
    map(float, os.getenv("ASPECT_RATIO_RANGE", "0.65,0.8").split(","))
)

pdf_render_dpi = int(os.getenv("PDF_RENDER_DPI", 150))
model_id = os.getenv("MODEL_ID", "25")

# Hidden smart-doc defaults (not exposed as API parameters).
DETAILED_DEFAULT = True
SIGN_URL_DEFAULT = False
UPLOAD_TYPE_DEFAULT = ""
REJECT_ASPECT_RATIO_DEFAULT = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    SingletonAiohttp.init_session()

    yield

    runner_dispatcher.release()
    await SingletonAiohttp.close_session()


app = FastAPI(
    title="Smart Document",
    version="2.0.1",
    description="Hiro-Smart-Doc API service: layout analysis + MOSS OCR for documents, PDFs and images.",
    contact={"name": "Hiro-Smart-Doc", "url": "https://github.com/patsnap/Hiro-Smart-Doc"},
    debug=False,
    swagger_ui_parameters={"persistAuthorization": True},
)


def custom_openapi() -> dict[str, Any]:
    """Build OpenAPI schema, accounting for the RD_API_PATH mount prefix."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Sub-app paths omit Mount prefix; Swagger joins servers[].url + path unless we set this.
    mount = (os.getenv("RD_API_PATH") or "").strip().rstrip("/")
    if mount:
        if not mount.startswith("/"):
            mount = "/" + mount
        openapi_schema["servers"] = [
            {
                "url": mount,
                # "description": "API base (RD_API_PATH); Try it out prepends this to each path.",
            }
        ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


# ==================== Pydantic Models ====================


class TextBox(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    # [x1, y1, x2, y2, confidence]
    bbox: Annotated[
        list[float],
        Len(min_length=5, max_length=5),
        AfterValidator(lambda b: list(map(lambda x: round(x, 4), b))),
    ]
    page: int
    type: str
    category: str
    content: str | None
    url: str | None = None
    model_id: str | None = None
    error: str | None = None


class Filter(BaseModel):
    main_text: bool = True

    table: bool = True
    chemical: bool = False
    equation: bool = True

    figure: bool = False
    supplemental_text: bool = False
    complex: bool = False
    others: bool = False

    # Workaround for using UploadFile with Pydantic model
    # https://stackoverflow.com/a/71439821
    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):  # type: ignore
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

def markdown_concat(parts: list[str]) -> str:
    text = ''''''
    for part in parts:
        json_part = json.loads(part)
        if json_part['category'] in ("main_text", "supplemental_text") and json_part['content'] is not None:
            text += json_part['content']+"\n"
        elif json_part['category'] in ("table", "equation") and json_part['content'] is not None:
            text += f"{json_part['content']}\n"
        elif json_part['category'] in ("figure", "others","complex","chemical") and json_part['url'] is not None:
            text += f"![{json_part['url']}]({json_part['url']})\n"

    return json.dumps({"markdown": text}, ensure_ascii=False)


# Sentinel for streaming page markdown when last_page=False (PDF multi-page).
# Chunk format: PAGE_MARKDOWN_SENTINEL + base64(page_markdown_json) + "\n"
PAGE_MARKDOWN_SENTINEL = "\x01SMART_DOC_PAGE_MD:"

# ==================== Image APIs ====================


async def smart_doc(
    image: MatLike,
    file_path: str,
    page_number: int,
    filter_options: dict[str, bool],
    ocr_filter_options: dict[str, bool],
    upload_filter_options: dict[str, bool],
    model_id: str,
    detailed: bool,
    sign_url: bool,
    upload_type: str,
    reject_aspect_ratio: bool,
    markdown: bool = False,
    last_page: bool = True,
    *,
    timing_out: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """When markdown=True and last_page=True, yields concatenated markdown once at end.
    When markdown=True and last_page=False, yields page markdown as sentinel chunk for caller to collect."""
    # Reject unsupported aspect ratio
    # aspect_ratio = get_aspect_ratio(*image.shape[1::-1])
    # if reject_aspect_ratio and not (
    #     aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]
    # ):
    #     if detailed:
    #         err_result = TextBox(
    #             bbox=[0, 0, 1, 1, 0],
    #             page=page_number,
    #             type="noise",
    #             category="others",
    #             content=None,
    #             url=None,
    #             error=f"Unsupported aspect ratio: {aspect_ratio}, "
    #             f"only support {aspect_ratio_range}",
    #         )
    #         yield err_result.model_dump_json() + "\n"
    #     return

    rec = StageRecorder("smart_doc", page=page_number, path=file_path)

    _bboxes = await runner_dispatcher.layout_inference(image, model_id)
    rec.mark("layout")
    bboxes, types, categories = runner_dispatcher.filter(_bboxes, filter_options, model_id)

    cropped_list = [c async for c in crop_image_bbox(image, bboxes)] if bboxes else []

    batch_entries: list[tuple[int, MatLike, str]] = [
        (i, cropped_list[i], categories[i])
        for i in range(len(bboxes))
        if categories[i] in ("table", "equation", "main_text", "supplemental_text")
        and ocr_filter_options[categories[i]]
    ]

    # Single MOSS batch over all OCR-eligible crops; results sliced by crop index.
    ocr_results_by_index: dict[int, str] = {}
    rec.mark("filter_crop_prep")
    if batch_entries:
        try:
            backend_payloads = [(img, cat) for _idx, img, cat in batch_entries]
            batch_out = await runner_dispatcher.moss_ocr_batch(
                backend_payloads,
                page=page_number,
                pdf_path=file_path,
            )
            for j, (idx, _img, _cat) in enumerate(batch_entries):
                if j < len(batch_out):
                    ocr_results_by_index[idx] = batch_out[j]
                else:
                    logger.error(
                        "Batch OCR length mismatch: expected %s results, got %s",
                        len(batch_entries),
                        len(batch_out),
                    )
                    ocr_results_by_index[idx] = ""
        except Exception as e:
            logger.error(f"Batch OCR error: {e}")
            for idx, _, _ in batch_entries:
                ocr_results_by_index[idx] = ""

    rec.mark("ocr")

    async def ocr(i: int, img: MatLike) -> str | None:
        # OCR — text only from the single batch response above (no per-crop dispatch).
        text, error = None, None
        if ocr_filter_options[categories[i]]:
            try:
                text = ocr_results_by_index.get(i)
                text = text.replace("\n", " ").strip() if text else ""
            except Exception as e:
                logger.error(f"OCR error: {e}")
                error = str(e)

        # Only return text if not detailed
        if not detailed:
            if text is None:
                return None
            return text + "\n"

        # Save cropped image to local storage; url is a full static URL.
        url = None
        if upload_filter_options[categories[i]]:
            rel_key = os.path.join(
                upload_prefix,
                upload_type,
                f"{now_time}-{file_path}/{page_number}_{i}.png",
            )
            image_bytes = save_image(img, ".png")
            await asyncio.to_thread(save_image_local, image_bytes, rel_key)
            # sign_url retained for API compatibility; local storage always
            # returns the same full static URL.
            url = image_url(rel_key)

        textbox = TextBox(
            bbox=bboxes[i],
            page=page_number,
            type=types[i],
            category=categories[i],
            content=text,
            url=url,
            model_id=model_id,
            error=error,
        )
        return textbox.model_dump_json() + "\n"

    # Parallel OCR
    now_time = now_time_str()
    coro_pool = CoroutinePool(num_workers=req_parallelism)

    async def iter_cropped() -> AsyncIterator[tuple[int, MatLike]]:
        for i, img in enumerate(cropped_list):
            yield (i, img)

    results = coro_pool.map(ocr, iter_cropped())
    markdown_parts: list[str] = [] if markdown else []
    async for result in results:
        if result is not None:
            if detailed and markdown:
                markdown_parts.append(result)
            yield result
    if detailed and markdown and markdown_parts:
        page_md = markdown_concat(markdown_parts)
        if last_page:
            yield page_md
        else:
            yield PAGE_MARKDOWN_SENTINEL + base64.b64encode(page_md.encode()).decode() + "\n"

    rec.mark("ocr_upload")
    payload = rec.finish()
    if timing_out is not None:
        timing_out.clear()
        timing_out.update(payload)
    else:
        rec.log_info(logger, payload=payload)


optional_str = "Optional, Example is default value."
filter_options_form = Form(
    Filter(),
    description=f"If return bboxes from this category. {optional_str}",
)
ocr_filter_options_form = Form(
    Filter(
        main_text=False,
        table=False,
        chemical=False,
        equation=False,
        figure=False,
        supplemental_text=False,
        complex=False,
        others=False,
    ),
    description=f"If do OCR on bboxes from this category. {optional_str}",
)
upload_filter_options_form = Form(
    Filter(main_text=False, figure=True),
    description="If upload image to object storage on bboxes from this category. "
    f"{optional_str}",
)
pdf_render_dpi_form = Form(
    pdf_render_dpi,
    description=f"How many DPI (dots per inch) when rendering PDF to image. {optional_str}",
)
markdown_form = Form(
    False,
    description="If true, append one more yield with concatenation of all results. "
    f"{optional_str}",
)
pdf_page_parallelism_form = Form(
    pdf_page_parallelism_default,
    ge=1,
    le=pdf_page_parallelism_max,
    description="Max PDF pages to process concurrently (capped by PDF_PAGE_PARALLELISM_MAX env). "
    f"{optional_str}",
)
last_page_form = Form(
    True,
    description="When markdown is true: if true, yield concatenated markdown at end of this page; "
    "if false, page markdown is streamed for collector (e.g. PDF uses this to merge all pages). "
    f"{optional_str}",
)


@app.post("/image/smart-doc", tags=["Smart Document"])
@log_error(logger)
async def image_smart_doc(
    image: UploadFile = File(...),
    filter_options: Filter = filter_options_form,
    ocr_filter_options: Filter = ocr_filter_options_form,
    upload_filter_options: Filter = upload_filter_options_form,
    markdown: bool = markdown_form,
    # last_page: bool = last_page_form,
) -> StreamingResponse:
    """Do full smart document process."""
    logger.info(f"Processing {image.filename}")
    image_bytes = image.file.read()
    results = smart_doc(
        load_image(image_bytes),
        f"{image.filename}-{hash_file(image_bytes)}",
        0,
        filter_options.model_dump(),
        ocr_filter_options.model_dump(),
        upload_filter_options.model_dump(),
        model_id,
        DETAILED_DEFAULT,
        SIGN_URL_DEFAULT,
        UPLOAD_TYPE_DEFAULT,
        REJECT_ASPECT_RATIO_DEFAULT,
        markdown,
        last_page=True,  # single image is always last page
    )
    return StreamingResponse(results)


async def _pdf_smart_doc_response(
    pdf_bytes: bytes,
    filename: str | None,
    filter_options: Filter = filter_options_form,
    ocr_filter_options: Filter = ocr_filter_options_form,
    upload_filter_options: Filter = upload_filter_options_form,
    pdf_render_dpi: int = pdf_render_dpi_form,
    pdf_page_parallelism: int = pdf_page_parallelism_form,
    model_id: str = model_id,
    detailed: bool = DETAILED_DEFAULT,
    sign_url: bool = SIGN_URL_DEFAULT,
    upload_type: str = UPLOAD_TYPE_DEFAULT,
    reject_aspect_ratio: bool = REJECT_ASPECT_RATIO_DEFAULT,
    markdown: bool = markdown_form,
) -> StreamingResponse:
    """Run the existing PDF page rendering and smart-doc pipeline."""
    filename = filename or "document.pdf"
    logger.info(f"Processing {filename}")

    pdf_path = f"{filename}-{hash_file(pdf_bytes)}"
    _filter_options = filter_options.model_dump()
    _ocr_filter_options = ocr_filter_options.model_dump()
    _upload_filter_options = upload_filter_options.model_dump()

    if not pdf_bytes:
        logger.error("empty request")
        return StreamingResponse(status_code=404, content="empty request")
    # Verify that the pdf bytes is a valid file
    try:
        pdf_page_count(pdf_bytes)
    except Exception as e:
        logger.error(f"{filename} file parsing failed! error msg: {e}")
        return StreamingResponse(
            status_code=406,
            content=f"file parsing failed! error msg: {e}",
        )

    effective_page_workers = max(
        1, min(pdf_page_parallelism, pdf_page_parallelism_max)
    )

    async def _gen() -> AsyncIterator[str]:
        pdf_rec = StageRecorder(
            "pdf_smart_doc",
            file=filename,
            pdf_path=pdf_path,
            pdf_page_parallelism=effective_page_workers,
        )
        n_pages = 0
        smart_doc_stage_ms_sum: dict[str, float] = {}
        pdf_render_ms_sum = 0.0
        end_logged = False
        try:
            n_pages = pdf_page_count(pdf_bytes)
            pdf_rec.mark("pdf_open")
            if n_pages == 0:
                return

            loop = asyncio.get_running_loop()
            page_done: dict[int, asyncio.Future[tuple[list[str], str | None]]] = {
                i: loop.create_future() for i in range(n_pages)
            }
            page_sem = asyncio.Semaphore(effective_page_workers)
            agg_lock = asyncio.Lock()

            async def _worker(page_number: int) -> None:
                nonlocal pdf_render_ms_sum, smart_doc_stage_ms_sum
                try:
                    async with page_sem:
                        page_rec = StageRecorder(
                            "pdf_smart_doc_page",
                            file=filename,
                            page=page_number,
                            total_pages=n_pages,
                            parallelism=effective_page_workers,
                        )
                        image = await asyncio.to_thread(
                            render_pdf_page_at_index,
                            pdf_bytes,
                            page_number,
                            pdf_render_dpi,
                        )
                        page_rec.mark("pdf_render")
                        chunks: list[str] = []
                        sentinel_md: str | None = None
                        page_timing: dict[str, Any] = {}
                        results = smart_doc(
                            image,
                            pdf_path,
                            page_number,
                            _filter_options,
                            _ocr_filter_options,
                            _upload_filter_options,
                            model_id,
                            detailed,
                            sign_url,
                            upload_type,
                            reject_aspect_ratio,
                            markdown,
                            last_page=False,
                            timing_out=page_timing,
                        )
                        async for r in results:
                            if r.startswith(PAGE_MARKDOWN_SENTINEL):
                                payload = r[len(PAGE_MARKDOWN_SENTINEL) :].strip()
                                sentinel_md = base64.b64decode(payload).decode()
                            else:
                                chunks.append(r)
                        page_rec.mark("smart_doc")
                        page_payload = page_rec.finish()
                        async with agg_lock:
                            pdf_ms = (
                                page_payload.get("stage_ms") or {}
                            ).get("pdf_render")
                            if pdf_ms is not None:
                                pdf_render_ms_sum += pdf_ms
                            for name, ms in (
                                page_timing.get("stage_ms") or {}
                            ).items():
                                smart_doc_stage_ms_sum[name] = (
                                    smart_doc_stage_ms_sum.get(name, 0.0) + ms
                                )
                        logger.info(
                            "latency_breakdown %s",
                            json.dumps(
                                {"pdf_page": page_payload, "smart_doc": page_timing},
                                ensure_ascii=False,
                            ),
                        )
                        page_done[page_number].set_result((chunks, sentinel_md))
                except Exception as e:
                    if not page_done[page_number].done():
                        page_done[page_number].set_exception(e)

            tasks = [asyncio.create_task(_worker(i)) for i in range(n_pages)]
            page_markdowns: list[str] = []
            try:
                for pn in range(n_pages):
                    chunks, md = await page_done[pn]
                    for c in chunks:
                        yield c
                    if md is not None:
                        page_markdowns.append(md)
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)

            if detailed and markdown and page_markdowns:
                combined = "\n\n".join(
                    json.loads(m)["markdown"] for m in page_markdowns
                )
                yield json.dumps({"markdown": combined}, ensure_ascii=False) + "\n"
            pdf_rec.mark("stream_complete")
            pdf_payload = pdf_rec.finish()
            pdf_payload["total_pages"] = n_pages
            pdf_payload["per_page_stages_sum_ms"] = {
                "pdf_render": round(pdf_render_ms_sum, 3),
                "smart_doc": {
                    k: round(v, 3) for k, v in sorted(smart_doc_stage_ms_sum.items())
                },
            }
            pdf_rec.log_info(logger, payload=pdf_payload, request_stage="end")
            end_logged = True
        finally:
            if not end_logged:
                pdf_payload = pdf_rec.finish()
                pdf_payload["total_pages"] = n_pages
                if n_pages > 0:
                    pdf_payload["per_page_stages_sum_ms"] = {
                        "pdf_render": round(pdf_render_ms_sum, 3),
                        "smart_doc": {
                            k: round(v, 3)
                            for k, v in sorted(smart_doc_stage_ms_sum.items())
                        },
                    }
                pdf_rec.log_info(logger, payload=pdf_payload, request_stage="end")

    return StreamingResponse(_gen())


@app.post("/pdf/smart-doc", tags=["Smart Document"])
@log_error(logger)
async def pdf_smart_doc(
    pdf: UploadFile = File(...),
    filter_options: Filter = filter_options_form,
    ocr_filter_options: Filter = ocr_filter_options_form,
    upload_filter_options: Filter = upload_filter_options_form,
    pdf_render_dpi: int = pdf_render_dpi_form,
    pdf_page_parallelism: int = pdf_page_parallelism_form,
    markdown: bool = markdown_form,
) -> StreamingResponse:
    """Do full smart document process."""
    pdf_bytes = pdf.file.read()
    return await _pdf_smart_doc_response(
        pdf_bytes,
        pdf.filename,
        filter_options,
        ocr_filter_options,
        upload_filter_options,
        pdf_render_dpi,
        pdf_page_parallelism,
        markdown=markdown,
    )


@app.post("/document/smart-doc", tags=["Smart Document"])
@log_error(logger)
async def document_smart_doc(
    document: UploadFile = File(...),
    filter_options: Filter = filter_options_form,
    ocr_filter_options: Filter = ocr_filter_options_form,
    upload_filter_options: Filter = upload_filter_options_form,
    pdf_render_dpi: int = pdf_render_dpi_form,
    pdf_page_parallelism: int = pdf_page_parallelism_form,
    markdown: bool = markdown_form,
) -> StreamingResponse:
    """Convert supported documents(doc, docx, ppt, pptx, xls, xlsx, odt, odp, ods, rtf) to PDF, then run the PDF smart-doc process."""
    logger.info(f"Processing {document.filename}")
    doc_rec = StageRecorder("document_smart_doc", file=document.filename)
    document_bytes = document.file.read()
    doc_rec.mark("file_read")
    try:
        converted = await convert_document_to_pdf(document_bytes, document.filename)
        doc_rec.mark("document_convert")
    except UnsupportedDocumentError as e:
        doc_rec.mark("document_convert_error")
        payload = doc_rec.finish()
        payload["status_code"] = 415
        payload["error"] = str(e)
        doc_rec.log_info(logger, payload=payload, request_stage="end")
        raise HTTPException(status_code=415, detail=str(e)) from e
    except DocumentConversionDisabledError as e:
        doc_rec.mark("document_convert_error")
        payload = doc_rec.finish()
        payload["status_code"] = 503
        payload["error"] = str(e)
        doc_rec.log_info(logger, payload=payload, request_stage="end")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except DocumentConversionError as e:
        doc_rec.mark("document_convert_error")
        payload = doc_rec.finish()
        payload["status_code"] = 406
        payload["error"] = str(e)
        doc_rec.log_info(logger, payload=payload, request_stage="end")
        raise HTTPException(status_code=406, detail=str(e)) from e

    doc_rec.meta.update(
        {
            "pdf_file": converted.pdf_filename,
            "source_extension": converted.source_extension,
            "converted": converted.converted,
            "pdf_bytes": len(converted.pdf_bytes),
        }
    )
    pdf_response = await _pdf_smart_doc_response(
        converted.pdf_bytes,
        converted.original_filename,
        filter_options,
        ocr_filter_options,
        upload_filter_options,
        pdf_render_dpi,
        pdf_page_parallelism,
        markdown=markdown,
    )
    doc_rec.mark("pdf_response_prepare")

    async def _gen() -> AsyncIterator[Any]:
        end_logged = False
        try:
            async for chunk in pdf_response.body_iterator:
                yield chunk
            doc_rec.mark("stream_complete")
            payload = doc_rec.finish()
            payload["status_code"] = pdf_response.status_code
            doc_rec.log_info(logger, payload=payload, request_stage="end")
            end_logged = True
        finally:
            if not end_logged:
                payload = doc_rec.finish()
                payload["status_code"] = pdf_response.status_code
                doc_rec.log_info(logger, payload=payload, request_stage="end")

    return StreamingResponse(
        _gen(),
        status_code=pdf_response.status_code,
        media_type=pdf_response.media_type,
        headers=dict(pdf_response.headers),
    )


@app.post("/document/convert-pdf", tags=["Smart Document"])
@log_error(logger)
async def document_convert_pdf(document: UploadFile = File(...)) -> StreamingResponse:
    """Convert a supported document(doc, docx, ppt, pptx, xls, xlsx, odt, odp, ods, rtf) to PDF and return the PDF for validation."""
    logger.info(f"Converting {document.filename} to PDF")
    document_bytes = document.file.read()
    try:
        converted = await convert_document_to_pdf(document_bytes, document.filename)
    except UnsupportedDocumentError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except DocumentConversionDisabledError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except DocumentConversionError as e:
        raise HTTPException(status_code=406, detail=str(e)) from e

    try:
        page_count = pdf_page_count(converted.pdf_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=406,
            detail=f"converted PDF parsing failed! error msg: {e}",
        ) from e

    headers = {
        "Content-Disposition": content_disposition_attachment(converted.pdf_filename),
        "X-Document-Converted": str(converted.converted).lower(),
        "X-Document-Source-Extension": converted.source_extension,
        "X-Document-PDF-Pages": str(page_count),
    }
    return StreamingResponse(
        io.BytesIO(converted.pdf_bytes),
        media_type="application/pdf",
        headers=headers,
    )
