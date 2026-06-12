"""Local image storage: replaces the former S3 upload + presign flow.

Images are written under ``LOCAL_IMAGE_DIR`` and exposed via the FastAPI
``StaticFiles`` mount at ``/static``. ``image_url`` returns a full URL built
from ``PUBLIC_BASE_URL``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("local_storage")

LOCAL_IMAGE_DIR = Path(os.getenv("LOCAL_IMAGE_DIR", "./output_images"))
STATIC_URL_PREFIX = os.getenv("STATIC_URL_PREFIX", "/static")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def save_image_local(image_bytes: bytes, rel_key: str) -> Path:
    """Write image bytes to ``LOCAL_IMAGE_DIR/rel_key``; returns the full path."""
    dest = LOCAL_IMAGE_DIR / rel_key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(image_bytes)
    return dest


def image_url(rel_key: str) -> str:
    """Full public URL for a stored image key."""
    prefix = STATIC_URL_PREFIX.strip("/")
    return f"{PUBLIC_BASE_URL}/{prefix}/{rel_key.lstrip('/')}"
