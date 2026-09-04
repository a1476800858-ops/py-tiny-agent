from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import Message, ToolDefinition, Usage

log = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def add_child(self, child: "Span") -> None:
        with self._lock:
            self.children.append(child)

    def add_attribute(self, key: str, value: Any) -> None:
        with self._lock:
            self.attributes[key] = value

    def end(self) -> None:
        with self._lock:
            if self.end_time is None:
                self.end_time = time.time()
                self.duration_ms = int((self.end_time - self.start_time) * 1000)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start_time)),
                "end_time": (
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.end_time))
                    if self.end_time is not None
                    else None
                ),
                "duration_ms": self.duration_ms,
                "attributes": dict(self.attributes),
                "children": [child.to_dict() for child in self.children],
            }


def start_span(parent: Span | None, name: str) -> Span:
    span = Span(name)
    if parent is not None:
        parent.add_child(span)
    return span


def export_trace_to_file(root_span: Span, work_dir: str, session_id: str) -> Path:
    trace_dir = Path(work_dir) / ".claw" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trace_{session_id}_{time.time_ns()}.json"
    path.write_text(json.dumps(root_span.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class Provider(Protocol):
    def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message: ...


PRICING_MODEL = {"glm-4.5-air": (0.15, 0.15)}


class CostTracker:
    def __init__(self, next_provider: Provider, model_name: str, session: Any) -> None:
        self.next_provider = next_provider
        self.model_name = model_name
        self.session = session

    def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        started = time.perf_counter()
        try:
            response = self.next_provider.generate(messages, available_tools)
        except Exception:
            log.exception("[Tracker] API 调用失败，耗时 %.3fs", time.perf_counter() - started)
            raise
        if response.usage is None:
            log.warning("[Tracker] API 调用完成，但未返回 Usage 数据")
            return response
        price = PRICING_MODEL.get(self.model_name, (0.0, 0.0))
        cost = (response.usage.prompt_tokens * price[0] + response.usage.completion_tokens * price[1]) / 1_000_000
        if self.session is not None:
            self.session.record_usage(response.usage.prompt_tokens, response.usage.completion_tokens, cost)
        log.info(
            "[Tracker] API 调用完成 | 输入: %d tk | 输出: %d tk | 花费: ¥%.6f",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            cost,
        )
        return response
