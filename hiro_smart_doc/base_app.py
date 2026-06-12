import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.routing import Mount
from fastapi.staticfiles import StaticFiles

from .common import utils
from .common.log import get_request_headers, set_request_headers
from .service import app as service_app
from .service import lifespan

api_path = os.getenv("RD_API_PATH", "")
api_logger = logging.getLogger("api")

STATIC_URL_PREFIX = os.getenv("STATIC_URL_PREFIX", "/static").rstrip("/") or "/static"
LOCAL_IMAGE_DIR = Path(os.getenv("LOCAL_IMAGE_DIR", "./output_images"))


app = FastAPI(
    routes=[Mount(api_path, service_app)],
    docs_url=None,
    lifespan=lifespan,
)

LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount(STATIC_URL_PREFIX, StaticFiles(directory=LOCAL_IMAGE_DIR), name="static")


@service_app.get(
    "/",
    tags=["Health Check"],
)
@service_app.get(
    "/health",
    tags=["Health Check"],
)
async def hello() -> str:
    return f"Hello world! Service {os.getenv('SERVICE-NAME', '')}, check /docs for more info."


@app.middleware("http")
async def after_requests(request: Request, call_next):  # type: ignore
    start_t = time.perf_counter()

    # Set contextVar for logging
    set_request_headers(request.headers)

    # Execute request
    response = await call_next(request)

    # Add response headers for service tracking
    headers = get_request_headers()
    response.headers.update(headers)

    # End logging
    api_logger.info(
        "after request",
        extra={
            "request_stage": "end",
            "uri": request.url,
            "status": response.status_code,
            "http_method": request.method,
            "total_time": utils.get_time_str(start_t),
        },
    )

    return response
