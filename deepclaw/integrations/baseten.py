"""Baseten-specific model helpers for DeepClaw."""

import json
from importlib import import_module

from langchain_core.messages import AIMessage, ToolMessage

from deepclaw.config import DeepClawConfig

BASETEN_PROVIDER = "baseten"


def load_chat_baseten_class():
    """Load LangChain's ChatBaseten adapter lazily."""
    try:
        module = import_module("langchain_baseten")
    except ImportError as exc:
        msg = (
            "Baseten models require langchain-baseten. Install it with `uv add langchain-baseten`."
        )
        raise RuntimeError(msg) from exc
    return module.ChatBaseten


def _sanitize_tool_message_content(content):
    """Convert OpenAI-incompatible structured tool output into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
                continue
            if block.get("type") == "image":
                mime_type = block.get("mime_type") or "image"
                text_parts.append(f"[image omitted: {mime_type}]")
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _normalize_tool_call_arguments(arguments):
    """Return parsed and serialized tool arguments, or (None, None) if unusable."""
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None, None
    if not isinstance(parsed, dict):
        return None, None
    return parsed, json.dumps(parsed, ensure_ascii=False)


def _sanitize_canonical_tool_calls(tool_calls) -> tuple[list[dict], list[dict], set[str]]:
    """Sanitize AIMessage.tool_calls into canonical and OpenAI-compatible shapes."""
    sanitized_canonical = []
    sanitized_openai = []
    dropped_tool_call_ids: set[str] = set()

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id")
        function_name = tool_call.get("name")
        if not tool_call_id or not function_name:
            if tool_call_id:
                dropped_tool_call_ids.add(str(tool_call_id))
            continue
        parsed_arguments, serialized_arguments = _normalize_tool_call_arguments(
            tool_call.get("args")
        )
        if parsed_arguments is None or serialized_arguments is None:
            dropped_tool_call_ids.add(str(tool_call_id))
            continue
        sanitized_canonical.append(
            {
                "name": function_name,
                "args": parsed_arguments,
                "id": tool_call_id,
                "type": tool_call.get("type", "tool_call"),
            }
        )
        sanitized_openai.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": serialized_arguments,
                },
            }
        )

    return sanitized_canonical, sanitized_openai, dropped_tool_call_ids


def _sanitize_openai_tool_calls(tool_calls) -> tuple[list[dict], list[dict], set[str]]:
    """Sanitize raw OpenAI-style tool call payloads from additional_kwargs."""
    sanitized_canonical = []
    sanitized_openai = []
    dropped_tool_call_ids: set[str] = set()

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id")
        function = tool_call.get("function")
        function_name = function.get("name") if isinstance(function, dict) else None
        if not tool_call_id or not isinstance(function, dict) or not function_name:
            if tool_call_id:
                dropped_tool_call_ids.add(str(tool_call_id))
            continue
        parsed_arguments, serialized_arguments = _normalize_tool_call_arguments(
            function.get("arguments")
        )
        if parsed_arguments is None or serialized_arguments is None:
            dropped_tool_call_ids.add(str(tool_call_id))
            continue
        sanitized_canonical.append(
            {
                "name": function_name,
                "args": parsed_arguments,
                "id": tool_call_id,
                "type": "tool_call",
            }
        )
        sanitized_openai.append(
            {
                "id": tool_call_id,
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": function_name,
                    "arguments": serialized_arguments,
                },
            }
        )

    return sanitized_canonical, sanitized_openai, dropped_tool_call_ids


def _sanitize_ai_message(message: AIMessage) -> tuple[AIMessage, set[str]]:
    """Drop malformed historical tool calls that OpenAI-compatible APIs reject."""
    dropped_tool_call_ids: set[str] = set()
    invalid_tool_calls = []

    if message.invalid_tool_calls:
        for tool_call in message.invalid_tool_calls:
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                dropped_tool_call_ids.add(str(tool_call_id))
    else:
        invalid_tool_calls = message.invalid_tool_calls

    additional_kwargs = dict(message.additional_kwargs)
    raw_openai_tool_calls = additional_kwargs.get("tool_calls")

    if message.tool_calls:
        sanitized_canonical, sanitized_openai, tool_call_drops = _sanitize_canonical_tool_calls(
            message.tool_calls
        )
    elif isinstance(raw_openai_tool_calls, list):
        sanitized_canonical, sanitized_openai, tool_call_drops = _sanitize_openai_tool_calls(
            raw_openai_tool_calls
        )
    else:
        sanitized_canonical, sanitized_openai, tool_call_drops = [], [], set()

    dropped_tool_call_ids.update(tool_call_drops)

    if sanitized_openai:
        additional_kwargs["tool_calls"] = sanitized_openai
    else:
        additional_kwargs.pop("tool_calls", None)

    if (
        sanitized_canonical == message.tool_calls
        and additional_kwargs == message.additional_kwargs
        and invalid_tool_calls == message.invalid_tool_calls
    ):
        return message, dropped_tool_call_ids
    return (
        message.model_copy(
            update={
                "tool_calls": sanitized_canonical,
                "additional_kwargs": additional_kwargs,
                "invalid_tool_calls": invalid_tool_calls,
            }
        ),
        dropped_tool_call_ids,
    )


def _sanitize_tool_messages(messages):
    """Normalize tool-related messages to OpenAI-compatible history."""
    sanitized_messages = []
    dropped_tool_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in dropped_tool_call_ids:
            continue
        if isinstance(message, AIMessage):
            sanitized_message, dropped_ids = _sanitize_ai_message(message)
            dropped_tool_call_ids = dropped_ids
            sanitized_messages.append(sanitized_message)
            continue
        if dropped_tool_call_ids and not isinstance(message, ToolMessage):
            dropped_tool_call_ids = set()
        if isinstance(message, ToolMessage) and not isinstance(message.content, str):
            sanitized_messages.append(
                message.model_copy(
                    update={"content": _sanitize_tool_message_content(message.content)}
                )
            )
        else:
            sanitized_messages.append(message)
    return sanitized_messages


def resolve_baseten_model(config: DeepClawConfig):
    """Resolve a baseten:* model spec into a wrapped ChatBaseten model."""
    model_spec = (config.model or "").strip()
    provider, separator, model_name = model_spec.partition(":")
    if separator == "" or provider != BASETEN_PROVIDER:
        return model_spec
    if not model_name:
        msg = "Baseten model name cannot be empty"
        raise ValueError(msg)

    chat_cls = load_chat_baseten_class()

    class WrappedChatBaseten(chat_cls):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            messages = self._convert_input(input_).to_messages()
            sanitized_messages = _sanitize_tool_messages(messages)
            return super()._get_request_payload(sanitized_messages, stop=stop, **kwargs)

    generation = config.generation
    kwargs = {
        "streaming": True,
        "disable_streaming": False,
        "stream_usage": True,
    }
    if model_name.startswith(("https://", "http://")):
        kwargs["model_url"] = model_name
    else:
        kwargs["model"] = model_name

    model_kwargs = {}
    if generation.temperature is not None:
        kwargs["temperature"] = generation.temperature
    if generation.max_tokens is not None:
        kwargs["max_tokens"] = generation.max_tokens
    if generation.top_p is not None:
        kwargs["top_p"] = generation.top_p
    if generation.repetition_penalty is not None:
        model_kwargs["repetition_penalty"] = generation.repetition_penalty
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    return WrappedChatBaseten(**kwargs)
