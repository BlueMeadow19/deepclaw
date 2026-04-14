import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepclaw import agent as agent_mod
from deepclaw.state_cache import invalidate_thread_state


class TestInvalidateThreadState:
    @pytest.mark.asyncio
    async def test_removes_cached_memory_and_skills_but_keeps_messages(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "messages": ["hello"],
                "memory_contents": {"/tmp/AGENTS.md": "cached"},
                "skills_metadata": [{"name": "hn-fetch"}],
            }
            await checkpointer.aput(config, checkpoint, {"step": 0}, {})

            changed = await invalidate_thread_state(checkpointer, "thread-1")
            latest = await checkpointer.aget_tuple(config)

        assert changed is True
        assert latest is not None
        assert latest.checkpoint["channel_values"] == {"messages": ["hello"]}

    @pytest.mark.asyncio
    async def test_returns_false_when_no_cached_context_keys_exist(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            config = {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}}
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"messages": ["hello"]}
            await checkpointer.aput(config, checkpoint, {"step": 0}, {})

            changed = await invalidate_thread_state(checkpointer, "thread-2")
            latest = await checkpointer.aget_tuple(config)

        assert changed is False
        assert latest is not None
        assert latest.checkpoint["channel_values"] == {"messages": ["hello"]}


class TestRuntimeContext:
    def test_includes_model_paths_and_guidance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_mod, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(agent_mod, "MEMORY_FILE", tmp_path / "AGENTS.md")

        context = agent_mod._build_runtime_context("anthropic:claude-opus")

        assert "Active model: anthropic:claude-opus" in context
        assert f"Memory file: {tmp_path / 'AGENTS.md'}" in context
        assert f"Local skills directory: {tmp_path / 'skills'}" in context
        assert "memory_add, memory_search, memory_replace, and memory_remove" in context
        assert (
            "Use skills_list to confirm what is installed and skill_view to read a skill" in context
        )
