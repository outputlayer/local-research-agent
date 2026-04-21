"""Bootstrap initial plan: JSON parse, seed validation, pipeline integration."""
import json

import pytest

# ── parse_bootstrap_json ────────────────────────────────────────────────────

def test_parse_strict_json():
    from lra.plan import parse_bootstrap_json
    out = parse_bootstrap_json(
        '{"topic_type":"theoretical","tasks":[{"title":"Sobolev inequalities in PAC-Bayes bounds","why":"core"}]}'
    )
    assert out is not None
    tt, tasks = out
    assert tt == "theoretical"
    assert len(tasks) == 1
    assert tasks[0]["title"].startswith("Sobolev")


def test_parse_wrapped_in_markdown_fence():
    from lra.plan import parse_bootstrap_json
    raw = '```json\n{"topic_type":"engineering","tasks":[]}\n```'
    out = parse_bootstrap_json(raw)
    assert out is not None and out[0] == "engineering"


def test_parse_with_llm_preamble():
    """LLM часто кладёт 'Вот JSON:' вокруг — парсер должен найти {...}."""
    from lra.plan import parse_bootstrap_json
    raw = 'Вот план:\n\n{"topic_type":"mixed","tasks":[{"title":"x x x x x x x x x x","why":"y"}]}\n\nГотово.'
    out = parse_bootstrap_json(raw)
    assert out is not None
    assert out[0] == "mixed"


def test_parse_garbage_returns_none():
    from lra.plan import parse_bootstrap_json
    assert parse_bootstrap_json("") is None
    assert parse_bootstrap_json("no json here") is None
    assert parse_bootstrap_json("{not valid json at all") is None


def test_parse_non_dict_root():
    from lra.plan import parse_bootstrap_json
    assert parse_bootstrap_json('["just", "an", "array"]') is None


# ── bootstrap_from_seeds ────────────────────────────────────────────────────

@pytest.fixture
def _isolated_plan(tmp_path, monkeypatch):
    from lra import plan as plan_mod
    monkeypatch.setattr(plan_mod, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan_mod, "PLAN_PATH", tmp_path / "plan.md")
    return tmp_path


def test_bootstrap_happy_path(_isolated_plan):
    from lra.plan import bootstrap_from_seeds, load
    seeds = [
        {"title": "Convergence rate of SGD under non-convexity", "why": "core theorem"},
        {"title": "Lower bounds via information-theoretic arguments", "why": "tight bounds"},
        {"title": "Relation to PAC-Bayes and generalization", "why": "bridge"},
        {"title": "Empirical validation of theoretical predictions", "why": "check"},
    ]
    p = bootstrap_from_seeds("SGD theory", seeds, topic_type="theoretical")
    assert p is not None
    # Plan written and loadable
    loaded = load()
    assert loaded is not None
    assert "theoretical" in loaded.root_goal
    assert len(loaded.tasks) == 4
    assert loaded.current_focus_id is not None
    # focus = first seed
    focus_task = loaded.get(loaded.current_focus_id)
    assert focus_task.title.startswith("Convergence rate")


def test_bootstrap_rejects_too_few_seeds(_isolated_plan):
    from lra.plan import bootstrap_from_seeds
    assert bootstrap_from_seeds("x", [], topic_type="mixed") is None
    assert bootstrap_from_seeds("x", [{"title": "valid title long enough", "why": "a"}],
                                topic_type="mixed") is None


def test_bootstrap_rejects_too_many_seeds(_isolated_plan):
    from lra.plan import bootstrap_from_seeds
    seeds = [{"title": f"task number {i} with enough length", "why": "w"} for i in range(9)]
    assert bootstrap_from_seeds("x", seeds, topic_type="mixed") is None


def test_bootstrap_filters_bad_entries(_isolated_plan):
    """Короткие/пустые titles отбрасываются; если валидных <3 — возвращаем None."""
    from lra.plan import bootstrap_from_seeds
    seeds = [
        {"title": "valid title one long enough", "why": "a"},
        {"title": "x", "why": "too short"},
        {"title": "", "why": "empty"},
        "not even a dict",
    ]
    assert bootstrap_from_seeds("x", seeds, topic_type="mixed") is None


def test_bootstrap_normalizes_unknown_topic_type(_isolated_plan):
    from lra.plan import bootstrap_from_seeds, load
    seeds = [
        {"title": "task one with enough chars", "why": "w"},
        {"title": "task two with enough chars", "why": "w"},
        {"title": "task three with enough chars", "why": "w"},
    ]
    p = bootstrap_from_seeds("q", seeds, topic_type="nonsense")
    assert p is not None
    assert "[mixed]" in load().root_goal  # fallback


# ── integration: _bootstrap_initial_plan (pipeline hook) ────────────────────

@pytest.fixture
def _isolated_all(tmp_path, monkeypatch):
    from lra import config, memory
    from lra import plan as plan_mod
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(memory, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(plan_mod, "PLAN_JSON_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(plan_mod, "PLAN_PATH", tmp_path / "plan.md")
    return tmp_path


def _fake_run_agent(response_content):
    """Builds a _run_agent stand-in that returns one assistant message."""
    def _fake(bot, messages, icon):
        return [{"role": "assistant", "content": response_content}]
    return _fake


def test_pipeline_bootstrap_applies_valid_json(_isolated_all, monkeypatch):
    from lra import pipeline
    from lra import plan as plan_mod
    # reset static plan first (как делает reset_research)
    plan_mod.reset("old static")

    payload = json.dumps({
        "topic_type": "engineering",
        "tasks": [
            {"title": "RAG retrieval: dense vs sparse vs hybrid", "why": "core"},
            {"title": "Vector store benchmarks (Faiss, Qdrant, Milvus)", "why": "impl"},
            {"title": "Chunking strategies and their effect on recall", "why": "tune"},
            {"title": "Failure modes: hallucination under low-recall", "why": "risk"},
        ],
    })
    monkeypatch.setattr(pipeline, "_run_agent", _fake_run_agent(payload))
    monkeypatch.setattr(pipeline, "build_bot", lambda *a, **kw: object())

    ok = pipeline._bootstrap_initial_plan("RAG architectures")
    assert ok is True
    loaded = plan_mod.load()
    assert "[engineering]" in loaded.root_goal
    assert any("RAG retrieval" in t.title for t in loaded.tasks)


def test_pipeline_bootstrap_falls_back_on_garbage(_isolated_all, monkeypatch):
    from lra import pipeline
    from lra import plan as plan_mod
    plan_mod.reset("topic")  # static baseline
    static_titles = {t.title for t in plan_mod.load().tasks}

    monkeypatch.setattr(pipeline, "_run_agent", _fake_run_agent("lol not json"))
    monkeypatch.setattr(pipeline, "build_bot", lambda *a, **kw: object())

    ok = pipeline._bootstrap_initial_plan("topic")
    assert ok is False
    # Статический план не тронут
    assert {t.title for t in plan_mod.load().tasks} == static_titles


def test_pipeline_bootstrap_survives_llm_exception(_isolated_all, monkeypatch):
    from lra import pipeline
    from lra import plan as plan_mod
    plan_mod.reset("topic")

    def _boom(*a, **kw):
        raise RuntimeError("mlx crashed")
    monkeypatch.setattr(pipeline, "build_bot", _boom)

    ok = pipeline._bootstrap_initial_plan("topic")
    assert ok is False
    # Статический план всё ещё есть
    assert plan_mod.load() is not None


def test_cfg_flag_default_on():
    """CFG['dynamic_initial_plan'] по дефолту считается True (guard in research_loop)."""
    from lra.config import CFG
    assert CFG.get("dynamic_initial_plan", True) is True
