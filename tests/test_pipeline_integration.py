"""Интеграционный тест research_loop с замоканным LLM.

Подменяет pipeline.build_bot и pipeline._run_agent, чтобы вместо реальных LLM-вызовов
каждая роль детерминированно производила нужные файловые артефакты через @register_tool.
Это ловит регрессии в оркестрации пайплайна без загрузки MLX.
"""
import pytest


def _patch_paths(monkeypatch, tmp_path):
    """Перенаправляет ВСЕ path-ссылки (config/memory/tools/pipeline/validator/metrics/kb/plan) в tmp_path."""
    from lra import config, kb, memory, metrics, pipeline, plan, research_memory, tools, validator
    paths = {
        "RESEARCH_DIR": tmp_path,
        "ARCHIVE_DIR": tmp_path / "archive",
        "DRAFT_PATH": tmp_path / "draft.md",
        "NOTES_PATH": tmp_path / "notes.md",
        "PLAN_PATH": tmp_path / "plan.md",
        "PLAN_JSON_PATH": tmp_path / "plan.json",
        "SYNTHESIS_PATH": tmp_path / "synthesis.md",
        "LESSONS_PATH": tmp_path / "lessons.md",
        "QUERYLOG_PATH": tmp_path / "querylog.md",
        "REJECTED_PATH": tmp_path / "rejected.jsonl",
        "METRICS_PATH": tmp_path / "metrics.json",
        "KB_PATH": tmp_path / "kb.jsonl",
        "RESEARCH_MEMORY_DIR": tmp_path / "memory",
    }
    for mod in (config, memory, tools, pipeline, validator, metrics, kb, plan, research_memory):
        for name, path in paths.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, path)


class StubBot:
    def __init__(self, role: str):
        self.role = role


def test_research_loop_end_to_end(tmp_path, monkeypatch):
    from lra import pipeline, tools

    _patch_paths(monkeypatch, tmp_path)

    # Фабрика ботов — распознаём роль по system_message
    def fake_build_bot(system_message, tool_names, max_tokens=None):
        msg = system_message
        if "researcher going deep" in msg:
            role = "explorer"
        elif "REPLANNER" in msg:
            role = "replanner"
        elif "SYNTHESIZER" in msg:
            role = "synthesizer"
        elif "FACT-CRITIC" in msg or "STRUCTURE-CRITIC" in msg or "scientific reviewer" in msg:
            role = "critic"
        elif "technical writer" in msg.lower():
            role = "writer"
        elif "compressor-filter" in msg:
            role = "compressor"
        else:
            role = "unknown"
        return StubBot(role)

    # Детерминированная симуляция tool-вызовов каждой ролью
    def fake_run_agent(bot, messages, icon):
        role = bot.role
        if role == "explorer":
            # Предварительно зарегистрируем id в kb — иначе pre-append verifier
            # (notes_strict) заблокирует запись с неизвестным arxiv-id.
            from lra import kb as _kb
            for _pid in ("2401.00001", "2401.00002"):
                _kb.add(_kb.Atom(id=_pid, kind="paper", topic="test", claim="seed"))
            tools.AppendNotes().call(
                {"content": "## [T1] обзорные архитектуры\n"
                            "[2401.00001] обзорная работа про архитектуры transformers\n"
                            "[2401.00002] анализ attention в обзорных архитектурах\n"
                            "[repo: foo/bar ★1200 Python] https://github.com/foo/bar\n"
                            "  модульный pipeline с plugin-архитектурой"})
            tools.AppendLessons().call({"content": "[iter 1] сработало: hf+gh; НЕ сработало: —; следующий: synth"})
        elif role == "replanner":
            # После первой итерации объявляем план завершённым → ранний стоп
            tools.WritePlan().call({"content": "# Plan: test\n\n[FOCUS] PLAN_COMPLETE\n\n"
                                                "## [TODO]\n\n## [DONE]\n- test: covered\n"})
        elif role == "synthesizer":
            tools.WriteSynthesis().call({"content":
                "## [BRIDGE]\n[2401.00001] <-> [2401.00002]: мост\n\n"
                "## [CONTRADICTION]\nконфликт между [2401.00001] и [2401.00002]\n\n"
                "## [GAP]\nпробел по [2401.00001]\n\n"
                "## [EXTRAPOLATION]\nтренд от [2401.00001] к [2401.00002]\n\n"
                "## [REUSE]\n[repo: foo/bar] — перенять pipeline-модульность\n\n"
                "## [TESTABLE]\nгипотеза по [2401.00001]: проверить\n"})
        elif role == "writer":
            tools.WriteDraft().call({"content":
                "# Отчёт\n\n## Введение\nо трансформерах [2401.00001] и внимании [2401.00002]\n\n"
                "## Novel Insights\n[BRIDGE] [CONTRADICTION] [GAP] [EXTRAPOLATION] [REUSE] [TESTABLE]\n\n"
                "## Practical Implementations\n- [repo: foo/bar] архитектурный ориентир\n\n"
                "## Источники\n1. [2401.00001]\n2. [2401.00002]\n"})
            return [{"role": "assistant", "content": "draft ready"}]
        elif role == "critic":
            return [{"role": "assistant", "content": "APPROVED"}]
        return [{"role": "assistant", "content": "done"}]

    monkeypatch.setattr(pipeline, "build_bot", fake_build_bot)
    monkeypatch.setattr(pipeline, "_run_agent", fake_run_agent)
    monkeypatch.setattr(pipeline, "prefetch_iteration",
                        lambda focus, **kw: {"hf": True, "gh": True, "elapsed": 0.0,
                                             "hf_cached": True, "gh_cached": True})

    # validator НЕ должен лезть в сеть — отключим внешние hf papers info вызовы
    from lra import validator
    monkeypatch.setattr(
        validator, "validate_draft_ids",
        lambda draft_text=None, notes_text=None, run_hf_info=True: (2, [], []),
    )

    pipeline.research_loop("test topic", depth=3, critic_rounds=1)

    # Проверяем артефакты
    assert (tmp_path / "notes.md").exists()
    notes = (tmp_path / "notes.md").read_text(encoding="utf-8")
    assert "[2401.00001]" in notes
    assert "[repo: foo/bar" in notes

    assert (tmp_path / "plan.md").exists()
    plan = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "PLAN_COMPLETE" in plan

    assert (tmp_path / "synthesis.md").exists()
    synth = (tmp_path / "synthesis.md").read_text(encoding="utf-8")
    for tag in ("[BRIDGE]", "[CONTRADICTION]", "[GAP]", "[EXTRAPOLATION]", "[REUSE]", "[TESTABLE]"):
        assert tag in synth

    assert (tmp_path / "draft.md").exists()
    draft = (tmp_path / "draft.md").read_text(encoding="utf-8")
    assert "Novel Insights" in draft
    assert "Practical Implementations" in draft
    for tag in ("[BRIDGE]", "[REUSE]"):
        assert tag in draft

    # metrics.json должен появиться и содержать ключевые поля
    import json
    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists(), "pipeline должен писать metrics.json"
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert data["query"] == "test topic"
    assert data["finished_at"] is not None
    assert len(data["iterations"]) >= 1
    assert data["iterations"][0]["new_arxiv_ids"] >= 1
    assert data["valid_ids"] == 2  # из нашего мока validate_draft_ids
    # specialized critics = 2 раунда (fact + structure), оба APPROVED
    assert len(data["critic_rounds"]) == 2
    assert all(r["approved"] for r in data["critic_rounds"])
    assert data["unique_cited_paper_ids"] >= 2
    assert data["source_diversity"] >= 3
    assert data["citation_coverage_ratio"] == 1.0

    memory_files = list((tmp_path / "memory").glob("*.md"))
    assert memory_files, "pipeline должен сохранять run-summary в research/memory/"


def test_research_loop_early_stop_on_plan_complete(tmp_path, monkeypatch):
    """PLAN_COMPLETE после 1-й итерации → дальше разворачивается к synth/writer."""
    from lra import pipeline, tools
    _patch_paths(monkeypatch, tmp_path)

    iters_called = []

    def fake_build_bot(system_message, tool_names, max_tokens=None):
        role = "explorer" if "исследователь" in system_message else (
            "replanner" if "РЕПЛАНЕР" in system_message else "other")
        return StubBot(role)

    def fake_run_agent(bot, messages, icon):
        if bot.role == "explorer":
            iters_called.append("explorer")
            from lra import kb as _kb
            _kb.add(_kb.Atom(id="2401.00001", kind="paper", topic="test", claim="seed"))
            tools.AppendNotes().call({"content": "[2401.00001] fact"})
            tools.AppendLessons().call({"content": "[iter] test"})
        elif bot.role == "replanner":
            tools.WritePlan().call({"content": "[FOCUS] PLAN_COMPLETE\n## [TODO]\n## [DONE]\n"})
        return [{"role": "assistant", "content": "done"}]

    monkeypatch.setattr(pipeline, "build_bot", fake_build_bot)
    monkeypatch.setattr(pipeline, "_run_agent", fake_run_agent)
    monkeypatch.setattr(pipeline, "prefetch_iteration",
                        lambda focus, **kw: {"hf": True, "gh": True, "elapsed": 0.0,
                                             "hf_cached": True, "gh_cached": True})
    monkeypatch.setattr("lra.validator.validate_draft_ids",
                        lambda **kw: (0, [], []))

    pipeline.research_loop("x", depth=10, critic_rounds=1)

    # depth=10, но PLAN_COMPLETE после 1-й → explorer должен быть вызван не более 1-2 раз
    assert len(iters_called) <= 2, f"expected early stop, got {len(iters_called)} iterations"
