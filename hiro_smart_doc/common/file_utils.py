import threading
from hashlib import blake2b
from typing import AsyncIterator

import cv2
import numpy as np
import pypdfium2 as pdfium  # type: ignore
from cv2.typing import MatLike
from PIL import Image

# PDFium is thread-incompatible (process-wide global state). Serialize every
# pdfium call through this lock so concurrent render workers can't corrupt it.
_PDFIUM_LOCK = threading.Lock()


def load_image(image_bytes: bytes) -> MatLike:
    return cv2.imdecode(
        np.asarray(bytearray(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR
    )


def get_image_size(image_bytes: bytes) -> tuple[int, int]:
    """(width, height)"""
    with Image.open(image_bytes) as img:
        return img.size


def get_aspect_ratio(width: int, height: int) -> float:
    """short_size / long_size, so value is between (0, 1)"""
    if width < height:
        return width / height
    return height / width


def hash_file(file_bytes: bytes) -> str:
    return blake2b(file_bytes, digest_size=4).hexdigest()


def save_image(image: MatLike, ext: str = ".png") -> bytes:
    match ext:
        case ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        case ".jpg" | ".jpeg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        case _:
            raise ValueError("Image type not supported")

    return cv2.imencode(ext, image, params)[1].tobytes()


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Open a PDF and return its page count. Raises on parse failure."""
    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(doc)
        finally:
            doc.close()


def render_pdf_page_at_index(pdf_bytes: bytes, page_index: int, dpi: int) -> MatLike:
    """Open the PDF, render one page to a BGR image (for use in a thread pool).

    PDFium is thread-incompatible, so the whole render is held under the
    process-wide lock.
    """
    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            page = doc[page_index]
            # pdfium scale is relative to 72 DPI.
            bitmap = page.render(scale=dpi / 72.0)
            try:
                pil_image = bitmap.to_pil()
                rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
            finally:
                bitmap.close()
                page.close()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            doc.close()


async def crop_image_bbox(
    image: MatLike, bboxes: list[list[float]]
) -> AsyncIterator[MatLike]:
    height, width = image.shape[:2]

    for bbox in bboxes:
        _x1, _y1, _x2, _y2 = bbox[:4]
        x1 = round(width * _x1)
        x2 = round(width * _x2)
        y1 = round(height * _y1)
        y2 = round(height * _y2)

        yield image[y1:y2, x1:x2, :]
