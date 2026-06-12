import asyncio
import io
import itertools
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


class DocumentConversionError(Exception):
    """Raised when a supported document cannot be converted to PDF."""


class UnsupportedDocumentError(DocumentConversionError):
    """Raised when the uploaded file type is not supported."""


class DocumentConversionDisabledError(DocumentConversionError):
    """Raised when Office->PDF conversion is not enabled on this deployment."""


@dataclass(frozen=True)
class ConvertedPdf:
    pdf_bytes: bytes
    original_filename: str
    pdf_filename: str
    source_extension: str
    converted: bool


PDF_EXTENSION = ".pdf"
DEFAULT_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".odt",
    ".odp",
    ".ods",
    ".rtf",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _supported_extensions() -> set[str]:
    raw = os.getenv("DOCUMENT_CONVERT_FORMATS", "").strip()
    if not raw:
        return DEFAULT_SUPPORTED_EXTENSIONS
    return {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (part.strip().lower() for part in raw.split(","))
        if ext
    }


def _safe_filename(filename: str | None) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    safe_stem = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9._-]+", "_", stem).strip("._")
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix).strip(".")
    if not safe_stem:
        safe_stem = "document"
    if safe_suffix:
        return f"{safe_stem}.{safe_suffix}"
    return safe_stem


def _temp_conversion_paths(workdir: Path, source_extension: str) -> tuple[Path, Path]:
    """ASCII-only temp paths for LibreOffice; it often fails on long/non-ASCII paths."""
    ext = source_extension if source_extension.startswith(".") else f".{source_extension}"
    return workdir / f"input{ext}", workdir / "output.pdf"


_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK"

_IMPORT_FILTER_BY_EXTENSION: dict[str, str] = {
    ".docx": "MS Word 2007 XML",
    ".doc": "MS Word 97",
    ".pptx": "Impress MS PowerPoint 2007 XML",
    ".ppt": "MS PowerPoint 97",
    ".xlsx": "Calc MS Excel 2007 XML",
    ".xls": "MS Excel 97",
    ".odt": "writer8",
    ".odp": "impress8",
    ".ods": "calc8",
    ".rtf": "Rich Text Format",
}


def _validate_office_bytes(file_bytes: bytes, source_extension: str) -> None:
    """Reject uploads whose content clearly does not match the declared extension."""
    if source_extension == ".docx":
        if not file_bytes.startswith(_ZIP_MAGIC):
            raise DocumentConversionError(
                "invalid docx: file is not a ZIP archive (wrong extension, corrupt upload, or not Office Open XML)"
            )
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise DocumentConversionError(
                        "invalid docx: missing word/document.xml (not a Word document package)"
                    )
        except zipfile.BadZipFile as e:
            raise DocumentConversionError(
                "invalid docx: corrupt ZIP archive"
            ) from e
    elif source_extension == ".doc" and not file_bytes.startswith(_OLE_MAGIC):
        if file_bytes.startswith(_ZIP_MAGIC):
            raise DocumentConversionError(
                "invalid doc: content looks like docx (ZIP); rename to .docx or re-save as .doc"
            )
        raise DocumentConversionError(
            "invalid doc: not a legacy OLE document (wrong extension or corrupt upload)"
        )


def _import_filter_for_extension(source_extension: str) -> str | None:
    return _IMPORT_FILTER_BY_EXTENSION.get(source_extension)


def _summarize_conversion_error(raw: str) -> str:
    if "Could not load document" in raw:
        return (
            "LibreOffice could not open the file. Common causes: corrupt document, "
            "password protection, extension does not match content, or unsupported features."
        )
    return raw


def _converter_commands() -> list[str]:
    configured = os.getenv("UNOCONVERTER_BIN", "").strip()
    if configured:
        return [configured]
    # unoserver docs mention both names across versions; try the common aliases.
    return ["unoconvert", "unoconverter"]


def _unoserver_endpoints() -> list[tuple[str, int]]:
    configured = os.getenv("UNOSERVER_ENDPOINTS", "").strip()
    if configured:
        endpoints: list[tuple[str, int]] = []
        for part in configured.split(","):
            host, separator, port = part.strip().rpartition(":")
            if not separator:
                continue
            endpoints.append((host or "127.0.0.1", int(port)))
        if endpoints:
            return endpoints

    host = os.getenv("UNOSERVER_HOST", "127.0.0.1")
    base_port = _env_int("UNOSERVER_PORT", 2003)
    instances = max(1, _env_int("UNOSERVER_INSTANCES", 2))
    port_step = _env_int("UNOSERVER_PORT_STEP", 10)
    return [(host, base_port + i * port_step) for i in range(instances)]


def _build_convert_command(
    command: str,
    input_path: Path,
    output_path: Path,
    *,
    host: str,
    port: int,
    input_filter: str | None = None,
) -> list[str]:
    args = [
        command,
        "--host",
        host,
        "--port",
        str(port),
        "--host-location",
        "local",
        "--convert-to",
        "pdf",
    ]
    if input_filter:
        args.extend(["--input-filter", input_filter])
    args.extend([str(input_path), str(output_path)])
    return args


_endpoints = _unoserver_endpoints()
_endpoint_cycle = itertools.cycle(_endpoints)
_endpoint_lock = asyncio.Lock()
_conversion_semaphore = asyncio.Semaphore(
    max(1, _env_int("DOCUMENT_CONVERT_MAX_CONCURRENCY", len(_endpoints)))
)


async def _next_unoserver_endpoint() -> tuple[str, int]:
    async with _endpoint_lock:
        return next(_endpoint_cycle)


async def convert_document_to_pdf(
    file_bytes: bytes,
    filename: str | None,
) -> ConvertedPdf:
    """Convert an uploaded office document(doc, docx, ppt, pptx, xls, xlsx, odt, odp, ods, rtf) to PDF bytes via a running unoserver."""
    if not file_bytes:
        raise DocumentConversionError("empty request")

    safe_name = _safe_filename(filename)
    source_extension = Path(safe_name).suffix.lower()
    if source_extension not in _supported_extensions():
        raise UnsupportedDocumentError(
            f"unsupported file type: {source_extension or '<empty>'}"
        )

    max_bytes = _env_int("DOCUMENT_CONVERT_MAX_BYTES", 50 * 1024 * 1024)
    if len(file_bytes) > max_bytes:
        raise DocumentConversionError(
            f"file too large: {len(file_bytes)} bytes exceeds {max_bytes} bytes"
        )

    pdf_filename = f"{Path(safe_name).stem}.pdf"
    if source_extension == PDF_EXTENSION:
        return ConvertedPdf(
            pdf_bytes=file_bytes,
            original_filename=safe_name,
            pdf_filename=pdf_filename,
            source_extension=source_extension,
            converted=False,
        )

    if not _env_bool("DOCUMENT_CONVERT_ENABLED", False):
        raise DocumentConversionDisabledError(
            "Office document conversion is disabled. Set DOCUMENT_CONVERT_ENABLED=true "
            "and run a LibreOffice unoserver (see README) to enable it."
        )

    _validate_office_bytes(file_bytes, source_extension)
    input_filter = _import_filter_for_extension(source_extension)

    timeout = _env_int("DOCUMENT_CONVERT_TIMEOUT", 60)
    async with _conversion_semaphore:
        with tempfile.TemporaryDirectory(prefix="smart-doc-convert-") as tmp:
            workdir = Path(tmp)
            input_path, output_path = _temp_conversion_paths(workdir, source_extension)
            input_path.write_bytes(file_bytes)
            host, port = await _next_unoserver_endpoint()

            last_error: str | None = None
            for command in _converter_commands():
                try:
                    process = await asyncio.create_subprocess_exec(
                        *_build_convert_command(
                            command,
                            input_path,
                            output_path,
                            host=host,
                            port=port,
                            input_filter=input_filter,
                        ),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError:
                    last_error = f"{command} command not found"
                    continue

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError as e:
                    process.kill()
                    await process.communicate()
                    raise DocumentConversionError(
                        f"document conversion timed out after {timeout}s"
                    ) from e

                if process.returncode == 0 and output_path.exists():
                    pdf_bytes = output_path.read_bytes()
                    if pdf_bytes:
                        return ConvertedPdf(
                            pdf_bytes=pdf_bytes,
                            original_filename=safe_name,
                            pdf_filename=pdf_filename,
                            source_extension=source_extension,
                            converted=True,
                        )

                raw_error = (
                    stderr.decode(errors="replace").strip()
                    or stdout.decode(errors="replace").strip()
                    or f"{command} exited with {process.returncode}"
                )
                last_error = _summarize_conversion_error(raw_error)
                break

    raise DocumentConversionError(f"document conversion failed: {last_error}")
