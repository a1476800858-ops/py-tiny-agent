from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any = field(default_factory=dict)

    def arguments_json(self) -> str:
        if isinstance(self.arguments, str):
            return self.arguments
        return json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    usage: Usage | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Message":
        usage = value.get("usage")
        return cls(
            role=str(value.get("role", "")),
            content=str(value.get("content") or ""),
            tool_calls=[
                ToolCall(
                    id=str(call.get("id", "")),
                    name=str(call.get("name", "")),
                    arguments=call.get("arguments", {}),
                )
                for call in value.get("tool_calls", [])
            ],
            tool_call_id=value.get("tool_call_id"),
            usage=Usage(**usage) if usage else None,
        )


@dataclass
class ToolResult:
    tool_call_id: str
    output: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
