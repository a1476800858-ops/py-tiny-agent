from __future__ import annotations

import os
from typing import Any

from .schema import Message, ToolCall, ToolDefinition, Usage, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER


class ProviderError(RuntimeError):
    pass


def _tool_schema(definition: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.input_schema,
        },
    }


def _openai_message(message: Message) -> dict[str, Any]:
    if message.role == ROLE_SYSTEM:
        return {"role": "system", "content": message.content}
    if message.role == ROLE_USER and message.tool_call_id:
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    if message.role == ROLE_USER:
        return {"role": "user", "content": message.content}
    result: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in message.tool_calls
        ]
    return result


class ZhipuOpenAIProvider:
    def __init__(self, model: str = "glm-4.5-air", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY", "")
        if not self.api_key:
            raise ProviderError("请设置 ZHIPU_API_KEY 环境变量")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("缺少 openai 依赖，请安装 py-tiny-claw[llm]") from exc
        self.client = OpenAI(api_key=self.api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")

    def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_openai_message(message) for message in messages],
        }
        if available_tools:
            kwargs["tools"] = [_tool_schema(tool) for tool in available_tools]
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI/Zhipu API 请求失败: {exc}") from exc
        if not response.choices:
            raise ProviderError("API 返回了空的 Choices")
        choice = response.choices[0].message
        calls = []
        for call in choice.tool_calls or []:
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        raw_usage = response.usage
        usage = (
            Usage(raw_usage.prompt_tokens, raw_usage.completion_tokens)
            if raw_usage is not None
            else None
        )
        return Message(ROLE_ASSISTANT, choice.content or "", calls, usage=usage)


class ZhipuClaudeProvider:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY", "")
        if not self.api_key:
            raise ProviderError("请设置 ZHIPU_API_KEY 环境变量")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError("缺少 anthropic 依赖，请安装 py-tiny-claw[llm]") from exc
        self.client = Anthropic(api_key=self.api_key, base_url="https://open.bigmodel.cn/api/anthropic")

    def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        system = ""
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == ROLE_SYSTEM:
                system = message.content
            elif message.role == ROLE_USER and message.tool_call_id:
                converted.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]})
            elif message.role == ROLE_USER:
                converted.append({"role": "user", "content": message.content})
            else:
                blocks: list[dict[str, Any]] = []
                blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    arguments = call.arguments
                    if isinstance(arguments, str):
                        import json

                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": arguments})
                converted.append({"role": "assistant", "content": blocks})
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": converted}
        if system:
            kwargs["system"] = system
        if available_tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in available_tools
            ]
        try:
            response = self.client.messages.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Claude/Zhipu API 请求失败: {exc}") from exc
        content = ""
        calls = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content += block.text
            elif getattr(block, "type", "") == "tool_use":
                calls.append(ToolCall(block.id, block.name, block.input))
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens) if response.usage else None
        return Message(ROLE_ASSISTANT, content, calls, usage=usage)


class MockProvider:
    """用于离线测试的按顺序返回响应的 Provider。"""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        self.calls.append((list(messages), list(available_tools)))
        if not self.responses:
            raise ProviderError("MockProvider 没有更多响应")
        return self.responses.pop(0)
