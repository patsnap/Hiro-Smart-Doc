import json
import logging
import os
import uuid
from contextvars import ContextVar

import concurrent_log_handler
from starlette.datastructures import Headers
from uvicorn.logging import DefaultFormatter

# https://inboard.bws.bio/logging#overriding-the-logging-config
# https://github.com/br3ndonland/inboard/blob/main/inboard/logging_conf.py


HEADER_KEYS_FOR_LOG = [
    "X-Amzn-Trace-Id",
    "X-Correlation-Id",
    "X-Client-Id",
    "X-User-Id",
    "X-Device-Id",
    "X-From",
    "X-Ai-Feature",
    "X-Result-Id",
]

DEFAULT_SERVICE_FROM = os.getenv("SERVICE_FROM", "hiro-smart-doc")

DOWNSTREAM_HEADER_DEFAULTS: dict[str, str | None] = {
    "X-Amzn-Trace-Id": "-",
    "X-Correlation-Id": None,
    "X-Client-Id": "-",
    "X-User-Id": "-",
    "X-Device-Id": "-",
    "X-From": DEFAULT_SERVICE_FROM,
    "X-Ai-Feature": "-",
    "X-Result-Id": "-",
}

_current_headers: ContextVar[dict[str, str]] = ContextVar("headers", default=dict())
_resolved_downstream_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "resolved_downstream_headers", default=None
)


def set_request_headers(headers: Headers) -> None:
    new_headers = {k: headers.get(k, "-") for k in HEADER_KEYS_FOR_LOG}
    _current_headers.set(new_headers)
    _resolved_downstream_headers.set(None)


def get_request_headers() -> dict[str, str]:
    return _current_headers.get()


def _resolve_header_value(
    key: str,
    raw: str,
    *,
    x_from_fallback: str | None,
) -> str:
    val = raw.strip()
    if val and val != "-":
        return val
    default = DOWNSTREAM_HEADER_DEFAULTS[key]
    if key == "X-Correlation-Id" and default is None:
        return str(uuid.uuid4())
    if key == "X-From":
        return x_from_fallback or str(default)
    return str(default)


def get_downstream_headers(*, x_from_fallback: str | None = None) -> dict[str, str]:
    """Build headers for downstream HTTP calls, filling defaults when missing or '-'."""
    cached = _resolved_downstream_headers.get()
    if cached is not None:
        if x_from_fallback and cached.get("X-From") in {
            "-",
            DEFAULT_SERVICE_FROM,
        }:
            return {**cached, "X-From": x_from_fallback}
        return cached

    headers = get_request_headers()
    resolved = {
        key: _resolve_header_value(
            key,
            headers.get(key, ""),
            x_from_fallback=x_from_fallback,
        )
        for key in HEADER_KEYS_FOR_LOG
    }
    _resolved_downstream_headers.set(resolved)
    return resolved


REQUIRED_LOG_KEYS = [
    "asctime",
    "request_stage",
    "levelname",
    "name",
    "X-Amzn-Trace-Id",
    "X-Correlation-Id",
    "X-Client-Id",
    "X-User-Id",
    "X-Device-Id",
    "X-From",
    "X-Ai-Feature",
    "X-Result-Id",
    "funcName",
    "step_time",
    "message",
    "api_platform_lib_version",
]

REQUIRED_CUSTOM_LOG_ITEMS = {
    "X-Amzn-Trace-Id": "-",
    "X-Correlation-Id": "-",
    "X-Client-Id": "-",
    "X-User-Id": "-",
    "X-Device-Id": "-",
    "X-From": "-",
    "X-Ai-Feature": "-",
    "X-Result-Id": "-",
    "step_time": "-",
    "total_time": "-",
    "status": "-",
    "request_stage": "during",
    "http_method": "POST",
    "uri": "-",
    "payload": "-",
    # called downstream service name
    "api_service_name": "-",
    "api_platform_lib_version": "0.0.00",
}

# not in this set, the key will be optional keys
# we support owverwrite these attrs, for details can check PatsnapFilter
RECORD_REQUIRED_KEYS = (
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    # "filename",  # Comment out filename to log it in extra fields
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    # "lineno",  # Comment out line number to log it in extra fields
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    # "process",  # Comment out process number to log it in extra fields
    "X-Correlation-Id",
    "X-Client-Id",
    "X-Amzn-Trace-Id",
    "X-User-Id",
    "X-Device-Id",
    "X-From",
    "X-Ai-Feature",
    "X-Result-Id",
    "step_time",
    "message",
    "asctime",
    "total_time",
    "status",
    "request_stage",
    "http_method",
    "uri",
    "payload",
    "api_service_name",
    "api_platform_lib_version",
)

_opt_logs_keys_str = os.getenv("OPTIONAL_LOG_KEYS", "").replace(" ", "")
OPTIONAL_LOG_KEYS = [k for k in _opt_logs_keys_str.split(",") if k]

API_REQUIRED_LOG_KEYS = [
    "asctime",
    "request_stage",
    "levelname",
    "name",
    "api_service_name",
    "X-Amzn-Trace-Id",
    "X-Correlation-Id",
    "X-Client-Id",
    "X-User-Id",
    "X-Device-Id",
    "X-From",
    "total_time",
    "http_method",
    "uri",
    "payload",
    "status",
    "funcName",
    "step_time",
    "message",
    "api_platform_lib_version",
]

# Blacklist `color_message` from uvicorn to prevent format errors
# Blacklist `headers` to prevent log it in extra fields
LOG_KEYS_BLACKLIST = ["color_message", "headers"]


class PatsnapFilter(logging.Filter):
    def __init__(self, opt_format: str | None = None) -> None:
        if opt_format is not None:
            assert opt_format in ["kvp", "json"], "opt_format must be 'kvp' or 'json'"
        self.opt_format = opt_format

    def _foramt_opt_logs(self, logs: dict[str, str], is_json: bool) -> str:
        """
        logs: dict, log items
        is_json: bool, is dump to json string
        """
        if not logs:
            return "-"
        if is_json:
            return json.dumps(logs)
        else:
            msg = " | ".join([f"{k}: {v}" for k, v in logs.items()])
            return msg

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "filtered") and record.filtered is True:
            return True

        headers = get_request_headers()
        for k, v in REQUIRED_CUSTOM_LOG_ITEMS.items():
            if not hasattr(record, k):
                if v == "-":
                    v = headers.get(k, "-")
                record.__setattr__(k, v)

        if record.payload != "-":  # type: ignore
            record.payload = json.dumps(record.payload)  # type: ignore

        # optf: kvp or json, kvp default
        is_json = False
        if self.opt_format is None:
            is_json = hasattr(record, "optf") and record.optf == "json"
        else:
            is_json = self.opt_format == "json"
        attrs = []
        for k, v in record.__dict__.items():
            if not k.startswith("_"):
                attrs.append(k)
            elif (nk := k[1:]) in RECORD_REQUIRED_KEYS:  # overwrite some values
                record.__setattr__(nk, v)
        log_items = {
            k: str(getattr(record, k))
            for k in attrs
            if k not in RECORD_REQUIRED_KEYS and k not in LOG_KEYS_BLACKLIST
        }
        _default_values = {k: "-" for k in OPTIONAL_LOG_KEYS if k not in log_items}
        log_items.update(_default_values)
        record.msg = self._foramt_opt_logs(log_items, is_json) + "] [" + record.msg

        record.filtered = True
        return True


service_logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose_fmt": {
            "()": DefaultFormatter,
            "format": " ".join([f"[%({k})s]" for k in REQUIRED_LOG_KEYS]),
        },
        "api_fmt": {
            "()": DefaultFormatter,
            "format": " ".join([f"[%({k})s]" for k in API_REQUIRED_LOG_KEYS]),
        },
    },
    "filters": {
        "patsnap_filter": {
            "()": PatsnapFilter,
            "opt_format": os.getenv("OPT_LOGS_FORMAT"),
        }
    },
    "handlers": {
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "verbose_fmt",
            "filters": ["patsnap_filter"],
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose_fmt",
            "filters": ["patsnap_filter"],
            "level": "INFO",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "formatter": "verbose_fmt",
            "filters": ["patsnap_filter"],
            "level": os.getenv("RD_LOG_LEVEL", "INFO"),
            "filename": os.getenv("RD_LOG_PATH", "/opt/logs/app.log"),
            "encoding": "utf-8",
            "maxBytes": 100 * 1024 * 1024,
            "backupCount": int(os.getenv("backupCount", 5)),
        },
        "api_console": {
            "class": "logging.StreamHandler",
            "formatter": "api_fmt",
            "filters": ["patsnap_filter"],
            "level": "INFO",
            "stream": "ext://sys.stderr",
        },
        "api_file": {
            "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "formatter": "api_fmt",
            "filters": ["patsnap_filter"],
            "level": os.getenv("RD_LOG_LEVEL", "INFO"),
            "filename": os.getenv("RD_LOG_PATH", "/opt/logs/app.log"),
            "encoding": "utf-8",
            "maxBytes": 100 * 1024 * 1024,
            "backupCount": int(os.getenv("backupCount", 5)),
        },
    },
    "root": {"handlers": ["console", "file"], "level": "INFO"},
    "loggers": {
        "api": {"handlers": ["api_console", "api_file"], "propagate": False},
        "gunicorn.access": {"handlers": ["access"], "propagate": False},
        "gunicorn.error": {"handlers": ["console", "file"], "propagate": False},
        # "fastapi": {"propagate": True},
        # "uvicorn": {"propagate": True},
        # "uvicorn.access": {"propagate": True},
        # "uvicorn.error": {"propagate": True},
        # "uvicorn.asgi": {"propagate": True},
    },
}
