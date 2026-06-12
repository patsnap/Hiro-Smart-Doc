from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


def latency_breakdown_enabled() -> bool:
    return os.getenv("LAYOUT_LATENCY_LOG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class StageRecorder:
    """
    Records durations between consecutive ``mark()`` calls (segment time).
    First ``mark()`` covers wall time from construction to that point.
    """

    def __init__(self, operation: str, **meta: Any) -> None:
        self.operation = operation
        self.meta = meta
        self._start = time.perf_counter()
        self._cursor = self._start
        self.segments: list[tuple[str, float]] = []

    @property
    def started_at(self) -> float:
        return self._start

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.segments.append((stage, (now - self._cursor) * 1000))
        self._cursor = now

    def finish(self) -> dict[str, Any]:
        wall_ms = (time.perf_counter() - self._start) * 1000
        stage_ms = {name: round(ms, 3) for name, ms in self.segments}
        stages_sum = round(sum(ms for _, ms in self.segments), 3)
        out: dict[str, Any] = {
            "operation": self.operation,
            "stage_ms": stage_ms,
            "stages_sum_ms": stages_sum,
            "wall_total_ms": round(wall_ms, 3),
        }
        out.update(self.meta)
        return out

    def log_info(
        self,
        log: logging.Logger,
        payload: dict[str, Any] | None = None,
        *,
        request_stage: str | None = None,
    ) -> None:
        if payload is None:
            payload = self.finish()
        msg = "latency_breakdown %s"
        body = json.dumps(payload, ensure_ascii=False)
        if request_stage is not None:
            log.info(msg, body, extra={"request_stage": request_stage})
        else:
            log.info(msg, body)
