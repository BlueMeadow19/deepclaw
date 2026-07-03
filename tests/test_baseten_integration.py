"""Tests for Baseten integration and agent wiring."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import invalid_tool_call

from deepclaw import agent as agent_mod
from deepclaw.config import DeepClawConfig
from deepclaw.integrations import resolve_provider_model
from deepclaw.integrations.baseten import BASETEN_PROVIDER, resolve_baseten_model


class TestResolveBasetenModel:
    def test_returns_plain_model_string_for_non_baseten(self):
        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6")

        resolved = resolve_baseten_model(config)

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_builds_chat_baseten_model_with_model_slug(self, monkeypatch):
        captured = {}

        class FakeChatBaseten:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        config = DeepClawConfig(model="baseten:moonshotai/Kimi-K2-Instruct-0905")
        config.generation.temperature = 0.25
        config.generation.max_tokens = 2048
        config.generation.top_p = 0.8
        config.generation.repetition_penalty = 1.1

        resolved = resolve_baseten_model(config)

        assert type(resolved).__name__ == "WrappedChatBaseten"
        assert captured == {
            "model": "moonshotai/Kimi-K2-Instruct-0905",
            "streaming": True,
            "disable_streaming": False,
            "stream_usage": True,
            "temperature": 0.25,
            "max_tokens": 2048,
            "top_p": 0.8,
            "model_kwargs": {"repetition_penalty": 1.1},
        }

    def test_builds_chat_baseten_model_with_dedicated_url(self, monkeypatch):
        captured = {}

        class FakeChatBaseten:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        config = DeepClawConfig(
            model="baseten:https://model-123.api.baseten.co/environments/production/sync/v1"
        )

        resolved = resolve_baseten_model(config)

        assert type(resolved).__name__ == "WrappedChatBaseten"
        assert captured["model_url"] == (
            "https://model-123.api.baseten.co/environments/production/sync/v1"
        )
        assert "model" not in captured

    def test_provider_dispatcher_routes_baseten(self, monkeypatch):
        fake_model = object()
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: fake_model
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(
            DeepClawConfig(model="baseten:moonshotai/Kimi-K2-Instruct-0905")
        )

        assert resolved is fake_model

    def test_provider_dispatcher_falls_back_to_raw_model(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: config.model.strip()
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(DeepClawConfig(model="anthropic:claude-sonnet-4-6"))

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_baseten_requires_model_name(self):
        config = DeepClawConfig(model=f"{BASETEN_PROVIDER}:")

        with pytest.raises(ValueError, match="Baseten model name cannot be empty"):
            resolve_baseten_model(config)

    def test_baseten_sanitizes_non_string_tool_message_content(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [{"content": msg.content} for msg in messages],
                    "stop": stop,
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            HumanMessage(content="hello"),
            ToolMessage(
                content=[
                    {"type": "image", "base64": "abc", "mime_type": "image/png"},
                    {"type": "text", "text": "caption text"},
                ],
                name="browser_screenshot",
                tool_call_id="call-1",
            ),
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert payload["messages"][0]["content"] == "hello"
        assert payload["messages"][1]["content"] == "[image omitted: image/png]\ncaption text"
        assert "abc" not in payload["messages"][1]["content"]

    def test_baseten_drops_invalid_tool_calls_and_orphaned_tool_messages(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [
                        {
                            "role": getattr(msg, "type", None),
                            "content": msg.content,
                            "tool_calls": msg.additional_kwargs.get("tool_calls"),
                            "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
                        }
                        for msg in messages
                    ]
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            HumanMessage(content="hello"),
            AIMessage(
                content="",
                invalid_tool_calls=[
                    invalid_tool_call(
                        name="execute",
                        args='{"command": "python3 -c "unterminated""}',
                        id="call-bad",
                        error="invalid json",
                    )
                ],
            ),
            ToolMessage(content="tool output", name="execute", tool_call_id="call-bad"),
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert len(payload["messages"]) == 2
        assert payload["messages"][1]["invalid_tool_calls"] == []

    def test_baseten_sanitizes_mixed_invalid_and_raw_tool_calls(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [
                        {
                            "tool_calls": msg.additional_kwargs.get("tool_calls"),
                            "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
                        }
                        for msg in messages
                    ]
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    invalid_tool_call(
                        name="execute",
                        args='{"command": "python3 -c "unterminated""}',
                        id="call-bad",
                        error="invalid json",
                    )
                ],
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call-good",
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": {"command": "echo hi"},
                            },
                        }
                    ]
                },
            ),
            ToolMessage(content="bad output", name="execute", tool_call_id="call-bad"),
            ToolMessage(content="good output", name="execute", tool_call_id="call-good"),
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["invalid_tool_calls"] == []
        assert (
            payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
            == '{"command": "echo hi"}'
        )

    def test_baseten_sanitizes_canonical_tool_calls(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [
                        {
                            "tool_calls": msg.additional_kwargs.get("tool_calls"),
                            "canonical_tool_calls": getattr(msg, "tool_calls", None),
                        }
                        for msg in messages
                    ]
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "echo hi"},
                        "id": "call-good",
                        "type": "tool_call",
                    }
                ],
            )
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert (
            payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
            == '{"command": "echo hi"}'
        )
        assert payload["messages"][0]["canonical_tool_calls"][0]["args"] == {"command": "echo hi"}

    def test_baseten_does_not_drop_later_reused_tool_call_id(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [
                        {
                            "role": getattr(msg, "type", None),
                            "content": msg.content,
                            "tool_call_id": getattr(msg, "tool_call_id", None),
                        }
                        for msg in messages
                    ]
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    invalid_tool_call(
                        name="execute",
                        args='{"command": "python3 -c "unterminated""}',
                        id="call-1",
                        error="invalid json",
                    )
                ],
            ),
            ToolMessage(content="bad output", name="execute", tool_call_id="call-1"),
            HumanMessage(content="try again"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "echo hi"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="good output", name="execute", tool_call_id="call-1"),
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert [message["content"] for message in payload["messages"]] == [
            "",
            "try again",
            "",
            "good output",
        ]

    def test_baseten_normalizes_raw_additional_tool_call_arguments(self, monkeypatch):
        class FakeChatBaseten:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _convert_input(self, input_):
                class _Input:
                    def __init__(self, messages):
                        self._messages = messages

                    def to_messages(self):
                        return self._messages

                return _Input(input_)

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                messages = self._convert_input(input_).to_messages()
                return {
                    "messages": [
                        {
                            "tool_calls": msg.additional_kwargs.get("tool_calls"),
                        }
                        for msg in messages
                    ]
                }

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        model = resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))

        messages = [
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": {"command": "echo hi"},
                            },
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": '{"command": "echo bye"}',
                            },
                        },
                    ]
                },
            )
        ]

        payload = model._get_request_payload(messages, stop=None)

        assert (
            payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
            == '{"command": "echo hi"}'
        )
        assert (
            payload["messages"][0]["tool_calls"][1]["function"]["arguments"]
            == '{"command": "echo bye"}'
        )

    def test_baseten_raises_helpful_error_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.baseten.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing langchain_baseten")),
        )

        with pytest.raises(RuntimeError, match="Baseten models require langchain-baseten"):
            resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))


class TestCreateAgentBaseten:
    def test_create_agent_passes_resolved_baseten_model_instance(self, tmp_path, monkeypatch):
        captured = {}
        fake_model = MagicMock(name="baseten-model")

        class FakeShellBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def execute(self, command, *, timeout=None):
                return None

        class FakeFilesystemBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeCompositeBackend:
            def __init__(self, *, default, routes):
                self.default = default
                self.routes = routes

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"

        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
        monkeypatch.setattr(agent_mod, "discover_tools", list)
        monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
        monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
        monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
        monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
        monkeypatch.setattr(
            agent_mod,
            "create_summarization_tool_middleware",
            lambda model, backend: ("compact-tool", model, backend),
            raising=False,
        )
        monkeypatch.setattr(agent_mod, "resolve_provider_model", lambda config: fake_model)

        config = DeepClawConfig(
            model="baseten:moonshotai/Kimi-K2-Instruct-0905", workspace_root=str(tmp_path)
        )
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert captured["model"] is fake_model
