import asyncio
import functools
import logging
import os
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Coroutine, TypeAlias, TypeVar
from urllib.parse import quote

from fastapi import BackgroundTasks, HTTPException, UploadFile


async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """
    Copy fastapi.UploadFile to a temp location on disk for further processing.
    """
    try:
        with open(destination, "wb") as buffer:
            await asyncio.to_thread(shutil.copyfileobj, upload_file.file, buffer)
    finally:
        upload_file.file.close()


def B2MB(size: int) -> float:
    """
    Convert Bytes to MegaBytes.
    """
    return round(size / (1024 * 1024), 2)


def content_disposition_attachment(filename: str) -> str:
    """Build a latin-1-safe Content-Disposition value (RFC 5987 for non-ASCII names)."""
    try:
        filename.encode("latin-1")
        return f'attachment; filename="{filename}"'
    except UnicodeEncodeError:
        ascii_fallback = filename.encode("ascii", "ignore").decode() or "download"
        if "." in filename and "." not in ascii_fallback:
            ascii_fallback = f"{ascii_fallback}{filename[filename.rfind('.'):]}"
        encoded = quote(filename)
        return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


def check_file_size(size: int, limit: int) -> None:
    """
    Check if exceed file size limitation.
    """
    if size > limit:  # Limit 20 MB
        raise HTTPException(
            status_code=413,
            detail=f"File size: {B2MB(size)} MB, hit limit of {B2MB(limit)} MB.",
        )


def tempfile_prepare(bg_tasks: BackgroundTasks, filename: str) -> tuple[str, str]:
    """
    Setup input/output location for file processing.
    """
    # Setup temp directory, register cleanup task
    temp_dir = tempfile.TemporaryDirectory()
    bg_tasks.add_task(temp_dir.cleanup)

    # Get paths for input/output files
    _split_name = os.path.splitext(filename)
    input_path = os.path.join(temp_dir.name, f"{_split_name[0]}_input{_split_name[1]}")
    output_path = os.path.join(temp_dir.name, filename)

    return input_path, output_path


def format_duration(seconds: float) -> str:
    """Format a duration in seconds (same adaptive rules as get_time_str)."""
    if seconds > 1:
        return f"{round(seconds, 2)}s"
    if seconds > 0.1:
        return f"{round(seconds * 1000, 1)}ms"
    return f"{round(seconds * 1000, 2)}ms"


def get_time_str(start_t: float) -> str:
    """
    Get time usage string, adaptive format.
    """
    return format_duration(time.perf_counter() - start_t)


def now_time_str() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")


T = TypeVar("T")


async def aenumerate(
    asequence: AsyncIterator[T], start: int = 0
) -> AsyncIterator[tuple[int, T]]:
    """Asynchronously enumerate an async iterator from a given start value"""
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1


R = TypeVar("R")
Func: TypeAlias = Callable[..., Coroutine[None, None, R]]


def log_error(logger: logging.Logger) -> Callable[[Func[R]], Func[R]]:
    """Log error stack on async functions"""

    def decorator(func: Func[R]) -> Func[R]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> R:  # type: ignore
            try:
                return await func(*args, **kwargs)
            except Exception:
                logger.error(traceback.format_exc())
                raise

        return wrapper

    return decorator


def retry(
    max_retry: int, logger: logging.Logger, operation: str
) -> Callable[[Func[R]], Func[R]]:
    """Retry operation for x times, only works on async functions"""

    def decorator(func: Func[R]) -> Func[R]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> R:  # type: ignore
            retry = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error during {operation}: {repr(e)}, {retry=}")
                    if (retry := retry + 1) > max_retry:
                        raise

        return wrapper

    return decorator


class CoroutinePool:
    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def map(
        self, func: Func[R], iterable: AsyncIterator[Any]
    ) -> AsyncIterator[R]:
        tasks = asyncio.Queue[asyncio.Task[R] | None](self.num_workers * 2)

        async def func_wrapper(_input: Any) -> R:
            async with self.semaphore:
                return await func(*_input)

        async def consume() -> None:
            async for _input in iterable:
                await tasks.put(tg.create_task(func_wrapper(_input)))
            await tasks.put(None)  # end signal

        async with asyncio.TaskGroup() as tg:
            tg.create_task(consume())

            while True:
                task = await tasks.get()
                if task is None:
                    return
                await task
                yield task.result()
