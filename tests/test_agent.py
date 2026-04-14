from deepagents.backends.filesystem import FilesystemBackend

from deepclaw import agent as agent_mod


class TestReloadingMemoryMiddleware:
    def test_reloads_memory_when_state_already_contains_cached_contents(self, tmp_path):
        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("first\n", encoding="utf-8")
        middleware = agent_mod.ReloadingMemoryMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[str(memory_file)],
        )

        initial = middleware.before_agent({}, None, None)
        assert initial == {"memory_contents": {str(memory_file): "first\n"}}

        memory_file.write_text("second\n", encoding="utf-8")
        refreshed = middleware.before_agent(
            {"memory_contents": {str(memory_file): "first\n"}},
            None,
            None,
        )

        assert refreshed == {"memory_contents": {str(memory_file): "second\n"}}


class TestReloadingSkillsMiddleware:
    def test_reloads_skills_when_new_skill_is_installed_mid_thread(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        hn_skill = skills_dir / "hn-fetch"
        hn_skill.mkdir()
        (hn_skill / "SKILL.md").write_text(
            "---\nname: hn-fetch\ndescription: Fetch Hacker News stories\n---\n# HN Fetch\n",
            encoding="utf-8",
        )

        middleware = agent_mod.ReloadingSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[str(skills_dir)],
        )

        initial = middleware.before_agent({}, None, None)
        assert [skill["name"] for skill in initial["skills_metadata"]] == ["hn-fetch"]

        web_skill = skills_dir / "web-fetch"
        web_skill.mkdir()
        (web_skill / "SKILL.md").write_text(
            "---\nname: web-fetch\ndescription: Fetch arbitrary web pages\n---\n# Web Fetch\n",
            encoding="utf-8",
        )

        refreshed = middleware.before_agent(
            {"skills_metadata": initial["skills_metadata"]},
            None,
            None,
        )

        assert [skill["name"] for skill in refreshed["skills_metadata"]] == [
            "hn-fetch",
            "web-fetch",
        ]


class TestRuntimeContext:
    def test_lists_installed_skills_and_tool_guidance(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("hn-fetch", "web-fetch"):
            skill_dir = skills_dir / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")

        monkeypatch.setattr(agent_mod, "SKILLS_DIR", skills_dir)
        monkeypatch.setattr(agent_mod, "MEMORY_FILE", tmp_path / "AGENTS.md")

        context = agent_mod._build_runtime_context("anthropic:claude-opus")

        assert "Installed local skills: hn-fetch, web-fetch" in context
        assert "memory_add, memory_search, memory_replace, and memory_remove" in context
        assert (
            "Use skills_list to confirm what is installed and skill_view to read a skill" in context
        )
