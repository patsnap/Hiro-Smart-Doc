<div align="center">

# Hiro-Smart-Doc

**Layout analysis + OCR API service for documents, PDFs and images — full smart-document processing for document pages.**

[English](README.md) | [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Layout Model](https://img.shields.io/badge/🤗%20Model-Hiro--Layout-yellow.svg)](https://huggingface.co/PatSnap/Hiro-Layout)

</div>

Hiro-Smart-Doc turns a document, PDF, or image into structured, reading-ordered
content. It runs an RT-DETR layout model to detect regions (text, tables,
equations, figures, chemical structures, …), sorts them into human reading
order (including multi-column pages), and runs OCR on each region to recover
text, HTML tables, and LaTeX formulas — optionally assembled into Markdown.

## Features

- **Layout analysis** — RT-DETR ONNX model detecting 25 region categories, with
  multi-column reading-order sorting and duplicate-box filtering.
- **Region OCR** — text, tables (HTML), and formulas (LaTeX) via the
  [MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) model served behind an
  OpenAI-compatible (vLLM) endpoint.
- **Multiple inputs** — single images, multi-page PDFs (rendered and processed
  per page with bounded concurrency), and Office documents (doc/docx/ppt/pptx/
  xls/xlsx/odt/odp/ods/rtf) via optional LibreOffice conversion.
- **Streaming API** — results stream back per region/page; an optional
  `markdown=true` flag returns a single concatenated Markdown document.
- **FastAPI service** — Swagger UI at `/docs`, plus an optional standalone
  Gradio UI for interactive exploration.

## Architecture

```
  PDF / image / Office doc
          │
          ▼
┌─────────────────────────────────────────────────┐
│  FastAPI service (hiro_smart_doc.base_app)
│
│  1. (Office) LibreOffice → PDF
│  2. Render page → image
│  3. Layout model (RT-DETR ONNX)  ──►  Hiro-Layout
│  4. Reading-order sort + filter
│  5. Region OCR  ──►  MOSS-OCR
│  6. Stream regions / assemble Markdown
└─────────────────────────────────────────────────┘
```

The two models are decoupled from this repo:

| Component   | What it does                          | Where it lives |
|-------------|---------------------------------------|----------------|
| Layout      | RT-DETR ONNX region detector          | [🤗 PatSnap/Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout) |
| OCR (MOSS)  | Text / table / formula recognition    | [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) |

The layout ONNX weights are **not** bundled in this repository; download them
from Hugging Face (see below). The OCR model runs as a separate
OpenAI-compatible service that this app calls over HTTP.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management
- **A running MOSS-OCR endpoint** (OpenAI-compatible / vLLM). See
  [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) for how to serve the
  model. Run it as a separate service from Hiro-Smart-Doc; for example, if
  Smart-Doc listens on `8000`, serve MOSS-OCR on `8088` and set
  `MOSS_VLLM_OCR_API=http://127.0.0.1:8088/v1`.
- **(Optional) An external LibreOffice unoserver**, only if you need to parse
  Office documents. Disabled by default; see
  [(Optional) Office document conversion](#optional-office-document-conversion).

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/patsnap/Hiro-Smart-Doc.git
cd Hiro-Smart-Doc

# Create the virtual environment and install dependencies (CPU onnxruntime)
uv sync

# For GPU inference instead of CPU:
uv sync --extra gpu

# Optional extras: Office-document conversion client, model download helper
uv sync --extra docconvert --extra download
```

> The `docconvert` extra installs the `unoconvert` **client** (to talk to a
> unoserver). The unoserver **server** must be deployed separately on
> LibreOffice's own Python — see
> [(Optional) Office document conversion](#optional-office-document-conversion).

## Download the layout model

The RT-DETR layout weights are hosted on Hugging Face at
[PatSnap/Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout). Fetch them
into `./layout_model/`:

```bash
# Using the helper script (requires the `download` extra)
uv run python scripts/download_models.py --models 25

# …or manually with the huggingface CLI
uv run huggingface-cli download PatSnap/Hiro-Layout RT-DETR_25.onnx \
    --local-dir ./layout_model
```

Files must follow the `RT-DETR_<MODEL_ID>.onnx` naming pattern (e.g.
`RT-DETR_25.onnx`), matching `MODEL_LIST` / `MODEL_ID` in your environment.

## Configuration

Copy the example environment file and adjust it:

```bash
cp .env.example .env
```

Key settings:

| Variable                | Description                                              | Default |
|-------------------------|----------------------------------------------------------|---------|
| `RD_INTERNAL_PORT`      | Port the API listens on                                  | `8000`  |
| `RD_API_PATH`           | Base path to mount the API under (empty = `/`)           | _empty_ |
| `MODEL_LIST`            | Comma-separated layout model ids to load                 | `25`    |
| `MODEL_ID`              | Default layout model id                                  | `25`    |
| `LAYOUT_MODEL_DIR`      | Directory holding `RT-DETR_<id>.onnx`                    | `./layout_model` |
| `RUNTIME_BACKEND`       | Inference backend                                        | `ONNX`  |
| `MOSS_VLLM_OCR_API`     | OpenAI-compatible MOSS-OCR endpoint (`.../v1`)           | `http://127.0.0.1:8088/v1` |
| `MOSS_VLLM_OCR_API_KEY` | API key for the OCR endpoint                             | `EMPTY` |
| `MOSS_VLLM_MODEL`       | OCR model name served by the endpoint                    | `moss-v1d6-0.3b` |
| `PDF_RENDER_DPI`        | DPI used when rendering PDF pages to images              | `150`   |
| `DOCUMENT_CONVERT_ENABLED` | Enable Office→PDF conversion (needs an external LibreOffice unoserver, see below) | `false` |
| `UNOSERVER_ENDPOINTS`   | unoserver address(es), `host:port`, comma-separated      | `127.0.0.1:2003` |
| `DOCUMENT_CONVERT_TIMEOUT` | Per-conversion timeout (seconds)                      | `60`    |
| `DOCUMENT_CONVERT_MAX_BYTES` | Max upload size in bytes                             | `52428800` |
| `DOCUMENT_CONVERT_MAX_CONCURRENCY` | Max concurrent conversions                    | number of endpoints |

## (Optional) Office document conversion

The `/document/*` endpoints and the Gradio document upload first convert
doc/docx/ppt/pptx/xls/xlsx/odt/odp/ods/rtf to PDF. This step is **disabled by
default** and is **not** built into the service — it relies on an external
[unoserver](https://github.com/unoconv/unoserver) (backed by LibreOffice). Skip
this section if you only process PDFs and images.

> **Why external:** high-fidelity conversion of these formats realistically
> requires LibreOffice, and its `uno` Python bindings (pyuno) are tied to the
> Python that ships with LibreOffice — they cannot be installed into this
> project's uv virtual environment (Python 3.12). So the unoserver **server**
> must be started with **LibreOffice's own / the system Python**, while this
> service only connects to it as a client over a socket.

**1. Install LibreOffice and unoserver:**

```bash
# Debian/Ubuntu: LibreOffice ships pyuno
sudo apt-get install -y libreoffice

# Server: install unoserver on the system Python that LibreOffice uses
# (NOT this project's venv); that Python must `python3 -c "import uno"` cleanly
sudo python3 -m pip install unoserver

# Client: install the unoconvert client into this project's venv
uv sync --extra docconvert
```

**2. Start unoserver** (long-running, listening on localhost):

```bash
# Start with the system Python that can import uno; port matches UNOSERVER_ENDPOINTS
python3 -m unoserver.server --interface 127.0.0.1 --port 2003 &
```

**3. Enable it and point the service at it** (`.env`):

```dotenv
DOCUMENT_CONVERT_ENABLED=true
UNOSERVER_ENDPOINTS=127.0.0.1:2003
# Scale out: run several unoserver ports, comma-separated
# UNOSERVER_ENDPOINTS=127.0.0.1:2003,127.0.0.1:2004
```

When disabled, the `/document/*` endpoints return `503` and the Gradio document
upload reports that conversion is unavailable; PDF and image pipelines are
unaffected.

## Running

Load the environment, then start the service. `uv run` automatically uses the
project virtual environment.

```bash
# export the variables from your .env (or use your preferred loader)
set -a && . ./.env && set +a

# Development (single process, auto-reload via uvicorn)
uv run uvicorn hiro_smart_doc.base_app:app --host 0.0.0.0 --port 8000

# Production (gunicorn + uvicorn workers)
uv run gunicorn --config gunicorn.conf.py hiro_smart_doc.base_app:app
```

Open the Swagger UI at <http://127.0.0.1:8000/docs>.

> If the logs show `POST /v1/chat/completions ... 404 Not Found`, double-check
> `MOSS_VLLM_OCR_API`: it must point at the MOSS-OCR/vLLM server (not the
> Hiro-Smart-Doc API) and include the `/v1` suffix.

### Gradio UI (optional)

An optional standalone interactive UI is available for exploring the pipeline
in a browser. It is not required to run the API service:

```bash
uv run python -m hiro_smart_doc.gradio_ui
# then open http://127.0.0.1:7860
```

## API

All endpoints accept `multipart/form-data` uploads and stream newline-delimited
JSON results.

| Method | Path                    | Description                                       |
|--------|-------------------------|---------------------------------------------------|
| POST   | `/image/smart-doc`      | Full pipeline on a single image                   |
| POST   | `/pdf/smart-doc`        | Full pipeline on a PDF (per-page, concurrent)     |
| POST   | `/document/smart-doc`   | Convert an Office doc to PDF, then run pipeline    |
| POST   | `/document/convert-pdf` | Convert an Office doc to PDF (returns the PDF)     |
| GET    | `/health`               | Health check                                      |

Common form fields: `filter_options` (which categories to return),
`ocr_filter_options` (which categories to OCR), and `markdown` (append a single
assembled Markdown document). See `/docs` for the full schema.

### Example

```bash
curl -X POST http://127.0.0.1:8000/pdf/smart-doc \
  -F "pdf=@paper.pdf" \
  -F 'ocr_filter_options={"main_text":true,"table":true,"equation":true}' \
  -F "markdown=true"
```

## Security

This service ships without authentication, and the local image store is served
publicly under `/static`. Before exposing it on an untrusted network, put it
behind a gateway or reverse proxy that enforces authentication, TLS, and rate
limiting.

## Related projects

- [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) — the OCR model used here
- [Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout) — the layout detection model

## License

Released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
attribution and trademark information.

Copyright (c) 2026 Patsnap. All rights reserved except as expressly licensed
under the applicable license terms.

Hiro-Smart-Doc, Patsnap, and any associated names, logos, product names,
service names, designs, and slogans are trademarks or registered trademarks of
Patsnap or its affiliates. No trademark license is granted under the open
source license or any model license unless expressly stated.
