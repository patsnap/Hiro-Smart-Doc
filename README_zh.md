<div align="center">

# Hiro-Smart-Doc

**面向文档、PDF 与图像的版面分析 + OCR 服务，对文档页面进行完整的智能文档处理。**

[English](README.md) | [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Layout Model](https://img.shields.io/badge/🤗%20Model-Hiro--Layout-yellow.svg)](https://huggingface.co/PatSnap/Hiro-Layout)

</div>

Hiro-Smart-Doc 将文档、PDF 或图像转换为结构化、按阅读顺序排列的内容。它使用
RT-DETR 版面模型检测页面区域（正文、表格、公式、图片、化学结构式等），将区域
按人类阅读顺序排序（支持多栏页面），再对每个区域执行 OCR，还原文本、HTML 表格
与 LaTeX 公式，并可选择拼接为 Markdown。

## 功能特性

- **版面分析** — RT-DETR ONNX 模型，检测 25 类区域，支持多栏阅读顺序排序与重复
  框过滤。
- **区域 OCR** — 通过 OpenAI 兼容（vLLM）接口调用
  [MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) 模型，识别文本、表格
  （HTML）与公式（LaTeX）。
- **多种输入** — 单张图像、多页 PDF（逐页渲染处理，并发可控），以及 Office 文档
  （doc/docx/ppt/pptx/xls/xlsx/odt/odp/ods/rtf，需可选的 LibreOffice 转换）。
- **流式接口** — 结果按区域/页面流式返回；可选 `markdown=true` 返回拼接后的单份
  Markdown 文档。
- **FastAPI 服务** — 提供 `/docs` Swagger UI，并附带可选的独立 Gradio 交互界面。

## 架构

```
  PDF / 图像 / Office 文档
          │
          ▼
┌─────────────────────────────────────────────────┐
│  FastAPI 服务 (hiro_smart_doc.base_app)
│
│  1. (Office) LibreOffice → PDF
│  2. 渲染页面 → 图像
│  3. 版面模型 (RT-DETR ONNX)  ──►  Hiro-Layout
│  4. 阅读顺序排序 + 过滤
│  5. 区域 OCR  ──►  MOSS-OCR
│  6. 流式返回区域 / 拼接 Markdown
└─────────────────────────────────────────────────┘
```

两个模型与本仓库解耦：

| 组件        | 作用                          | 位置 |
|-------------|-------------------------------|------|
| 版面 Layout | RT-DETR ONNX 区域检测器        | [🤗 PatSnap/Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout) |
| OCR (MOSS)  | 文本 / 表格 / 公式识别         | [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) |

版面 ONNX 权重**不**包含在本仓库中，请从 Hugging Face 下载（见下文）。OCR 模型
作为独立的 OpenAI 兼容服务运行，本应用通过 HTTP 调用。

## 环境要求

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** 用于环境与依赖管理
- **可访问的 MOSS-OCR 接口**（OpenAI 兼容 / vLLM）。部署方式参见
  [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR)。
- **（可选）外部 LibreOffice unoserver**，仅在需要解析 Office 文档时使用。
  默认关闭，部署方式见下方[（可选）Office 文档转换](#可选office-文档转换)。

## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后：

```bash
git clone https://github.com/patsnap/Hiro-Smart-Doc.git
cd Hiro-Smart-Doc

# 创建虚拟环境并安装依赖（CPU 版 onnxruntime）
uv sync

# 如需 GPU 推理：
uv sync --extra gpu

# 可选附加项：Office 文档转换客户端、模型下载工具
uv sync --extra docconvert --extra download
```

> `docconvert` 附加项装的是 `unoconvert` **客户端**（连接 unoserver 用）。
> unoserver **服务端**需用 LibreOffice 自带的 Python 单独部署，详见下方
> [（可选）Office 文档转换](#可选office-文档转换)。

## 下载版面模型

RT-DETR 版面权重托管在 Hugging Face 的
[PatSnap/Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout)，下载到
`./layout_model/`：

```bash
# 使用辅助脚本（需安装 download 附加项）
uv run python scripts/download_models.py --models 25

# 或使用 huggingface CLI 手动下载
uv run huggingface-cli download PatSnap/Hiro-Layout RT-DETR_25.onnx \
    --local-dir ./layout_model
```

文件名需遵循 `RT-DETR_<MODEL_ID>.onnx` 命名规则（例如 `RT-DETR_25.onnx`），
与环境变量中的 `MODEL_LIST` / `MODEL_ID` 对应。

## 配置

复制示例环境文件并按需修改：

```bash
cp .env.example .env
```

主要配置项：

| 变量                    | 说明                                              | 默认值 |
|-------------------------|---------------------------------------------------|--------|
| `RD_INTERNAL_PORT`      | API 监听端口                                      | `8000` |
| `RD_API_PATH`           | API 挂载的基础路径（为空则为 `/`）                | _空_   |
| `MODEL_LIST`            | 要加载的版面模型 id，逗号分隔                     | `25`   |
| `MODEL_ID`              | 默认版面模型 id                                   | `25`   |
| `LAYOUT_MODEL_DIR`      | 存放 `RT-DETR_<id>.onnx` 的目录                  | `./layout_model` |
| `RUNTIME_BACKEND`       | 推理后端                                          | `ONNX` |
| `MOSS_VLLM_OCR_API`     | OpenAI 兼容的 MOSS-OCR 接口（`.../v1`）          | `http://127.0.0.1:8000/v1` |
| `MOSS_VLLM_OCR_API_KEY` | OCR 接口的 API key                               | `EMPTY` |
| `MOSS_VLLM_MODEL`       | 接口提供的 OCR 模型名称                           | `moss-v1d6-0.3b` |
| `PDF_RENDER_DPI`        | PDF 页面渲染为图像的 DPI                          | `150`  |
| `DOCUMENT_CONVERT_ENABLED` | 是否启用 Office→PDF 转换（需外部 LibreOffice unoserver，见下文） | `false` |
| `UNOSERVER_ENDPOINTS`   | unoserver 地址，`host:port`，多个用逗号分隔        | `127.0.0.1:2003` |
| `DOCUMENT_CONVERT_TIMEOUT` | 单次转换超时（秒）                             | `60`   |
| `DOCUMENT_CONVERT_MAX_BYTES` | 允许上传的文档最大字节数                      | `52428800` |
| `DOCUMENT_CONVERT_MAX_CONCURRENCY` | 转换并发上限                            | unoserver 端点数 |

## （可选）Office 文档转换

`/document/*` 接口及 Gradio 的文档上传，需要把 doc/docx/ppt/pptx/xls/xlsx/odt/
odp/ods/rtf 先转成 PDF。这一步**默认关闭**，且**不**内置于本服务——它依赖一个
外部的 [unoserver](https://github.com/unoconv/unoserver)（基于 LibreOffice）。只
处理 PDF 和图像时可忽略本节。

> **为什么是外部服务：** 高保真转换这套格式实际只有 LibreOffice 能做，而它的
> `uno` Python 绑定（pyuno）与 LibreOffice 自带的 Python 版本强绑定，无法装进本项目
> 的 uv 虚拟环境（Python 3.12）。因此 unoserver 需用 **LibreOffice 自带/系统对应的
> Python** 单独启动；本服务只作为客户端通过 socket 连接它。

**1. 安装 LibreOffice 与 unoserver**：

```bash
# Debian/Ubuntu：LibreOffice 自带 pyuno
sudo apt-get install -y libreoffice

# 服务端：用 LibreOffice 对应的系统 Python 安装 unoserver（不要装进本项目 venv）
# 该 Python 必须能 `python3 -c "import uno"` 成功
sudo python3 -m pip install unoserver

# 客户端：本项目 venv 安装 unoconvert 客户端（连接 unoserver）
uv sync --extra docconvert
```

**2. 启动 unoserver**（常驻，监听本机端口）：

```bash
# 用能 import uno 的系统 Python 启动；端口与 UNOSERVER_ENDPOINTS 对应
python3 -m unoserver.server --interface 127.0.0.1 --port 2003 &
```

**3. 在本项目启用并指向它**（`.env`）：

```dotenv
DOCUMENT_CONVERT_ENABLED=true
UNOSERVER_ENDPOINTS=127.0.0.1:2003
# 横向扩展：多开几个 unoserver 端口，用逗号分隔
# UNOSERVER_ENDPOINTS=127.0.0.1:2003,127.0.0.1:2004
```

未启用时，`/document/*` 接口返回 `503`，Gradio 上传文档会提示转换不可用；PDF 与
图像流程不受影响。

## 运行

加载环境变量后启动服务。`uv run` 会自动使用项目虚拟环境。

```bash
# 从 .env 导出变量（或使用你习惯的加载方式）
set -a && . ./.env && set +a

# 开发模式（单进程，uvicorn 自动重载）
uv run uvicorn hiro_smart_doc.base_app:app --host 0.0.0.0 --port 8000

# 生产模式（gunicorn + uvicorn worker）
uv run gunicorn --config gunicorn.conf.py hiro_smart_doc.base_app:app
```

访问 Swagger UI：<http://127.0.0.1:8000/docs>。

### Gradio 界面（可选）

附带可选的独立交互式界面，用于在浏览器中探索流程，运行 API 服务时并非必需：

```bash
uv run python -m hiro_smart_doc.gradio_ui
# 然后打开 http://127.0.0.1:7860
```

## API

所有接口均接收 `multipart/form-data` 上传，并以换行分隔的 JSON 流式返回结果。

| 方法   | 路径                    | 说明                                              |
|--------|-------------------------|---------------------------------------------------|
| POST   | `/image/smart-doc`      | 对单张图像执行完整流程                            |
| POST   | `/pdf/smart-doc`        | 对 PDF 执行完整流程（逐页并发）                   |
| POST   | `/document/smart-doc`   | 将 Office 文档转为 PDF 后执行流程                 |
| POST   | `/document/convert-pdf` | 将 Office 文档转为 PDF 并返回 PDF                 |
| GET    | `/health`               | 健康检查                                          |

常用表单字段：`filter_options`（返回哪些类别）、`ocr_filter_options`（对哪些类别
执行 OCR）、`markdown`（追加拼接后的单份 Markdown 文档）。完整参数见 `/docs`。

### 示例

```bash
curl -X POST http://127.0.0.1:8000/pdf/smart-doc \
  -F "pdf=@paper.pdf" \
  -F 'ocr_filter_options={"main_text":true,"table":true,"equation":true}' \
  -F "markdown=true"
```

## 安全

本服务不含任何鉴权，且本地图像存储通过 `/static`
公开提供。在将其暴露到不可信网络之前，请置于具备鉴权、TLS 与限流能力的网关或
反向代理之后。

## 相关项目

- [Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR) — 本项目使用的 OCR 模型
- [Hiro-Layout](https://huggingface.co/PatSnap/Hiro-Layout) — 版面检测模型

## 许可证

基于 [Apache License 2.0](LICENSE) 发布。归属与商标信息见 [NOTICE](NOTICE)。

版权所有 (c) 2026 Patsnap。除非依据适用的许可条款明确授予，否则保留所有权利。

Hiro-Smart-Doc、Patsnap 以及任何相关名称、标识、产品名称、服务名称、设计与标语，
均为 Patsnap 或其关联公司的商标或注册商标。除非另有明确说明，开源许可证或任何
模型许可证均不授予任何商标许可。
