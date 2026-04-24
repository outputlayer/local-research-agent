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
    tt, tasks, vocab = out
    assert tt == "theoretical"
    assert len(tasks) == 1
    assert tasks[0]["title"].startswith("Sobolev")
    assert vocab == []  # vocab missing in JSON → empty list


def test_parse_with_core_vocabulary():
    from lra.plan import parse_bootstrap_json
    raw = (
        '{"topic_type":"mixed",'
        '"core_vocabulary":["jamming","ELINT","pulse deinterleaving","ESM"],'
        '"tasks":[{"title":"LPI radar detection under noise","why":"core"}]}'
    )
    out = parse_bootstrap_json(raw)
    assert out is not None
    tt, tasks, vocab = out
    assert tt == "mixed"
    assert vocab == ["jamming", "ELINT", "pulse deinterleaving", "ESM"]


def test_parse_core_vocabulary_ignores_non_strings():
    from lra.plan import parse_bootstrap_json
    raw = (
        '{"topic_type":"mixed",'
        '"core_vocabulary":["jamming",123,null,"ESM",{"x":1}],'
        '"tasks":[]}'
    )
    out = parse_bootstrap_json(raw)
    assert out is not None
    _, _, vocab = out
    assert vocab == ["jamming", "ESM"]


def test_parse_wrapped_in_markdown_fence():
    from lra.plan import parse_bootstrap_json
    raw = '```json\n{"topic_type":"engineering","tasks":[]}\n```'
    out = parse_bootstrap_json(raw)
    assert out is not None and out[0] == "engineering"


def test_parse_with_llm_preamble():
    """The LLM often wraps output with 'Here is JSON:' — the parser must locate {...}."""
    from lra.plan import parse_bootstrap_json
    raw = 'Here is the plan:\n\n{"topic_type":"mixed","tasks":[{"title":"x x x x x x x x x x","why":"y"}]}\n\nDone.'
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
    """Short/empty titles dropped; if valid <3 — return None."""
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


def test_bootstrap_sanitizes_core_vocabulary(_isolated_plan):
    """len 4-40, dedupe case-insensitive, cap 15, non-strings dropped."""
    from lra.plan import bootstrap_from_seeds, load
    seeds = [
        {"title": "task one with enough chars", "why": "w"},
        {"title": "task two with enough chars", "why": "w"},
        {"title": "task three with enough chars", "why": "w"},
    ]
    vocab = [
        "jamming",        # ok
        "x",              # too short (1 char)
        "a" * 50,         # too long
        "ELINT",          # ok
        "elint",          # dup (case-insensitive)
        "   ESM   ",      # ok (strip, 3 chars — threshold 3)
        123,              # not a string
        "",               # empty
    ] + [f"term{i}abcd" for i in range(20)]  # 20 → cap at 15
    p = bootstrap_from_seeds("q", seeds, topic_type="mixed", core_vocabulary=vocab)
    assert p is not None
    loaded = load()
    assert loaded.core_vocabulary[0] == "jamming"
    assert "ELINT" in loaded.core_vocabulary
    assert "ESM" in loaded.core_vocabulary
    # no duplicates
    lowers = [t.lower() for t in loaded.core_vocabulary]
    assert len(lowers) == len(set(lowers))
    # cap 15
    assert len(loaded.core_vocabulary) <= 15


def test_render_md_emits_core_vocabulary_line(_isolated_plan, tmp_path):
    from lra.plan import bootstrap_from_seeds
    seeds = [
        {"title": "task one with enough chars", "why": "w"},
        {"title": "task two with enough chars", "why": "w"},
        {"title": "task three with enough chars", "why": "w"},
    ]
    bootstrap_from_seeds(
        "EW topic", seeds, topic_type="mixed",
        core_vocabulary=["jamming", "ELINT", "ESM"],
    )
    md = (tmp_path / "plan.md").read_text()
    assert "**Core vocabulary:** jamming, ELINT, ESM" in md
    # line goes AFTER the header, BEFORE focus-line
    header_pos = md.index("# Plan:")
    vocab_pos = md.index("**Core vocabulary:**")
    focus_pos = md.index("[FOCUS]")
    assert header_pos < vocab_pos < focus_pos


def test_render_md_no_vocab_line_when_empty(_isolated_plan, tmp_path):
    from lra.plan import bootstrap_from_seeds
    seeds = [
        {"title": "task one with enough chars", "why": "w"},
        {"title": "task two with enough chars", "why": "w"},
        {"title": "task three with enough chars", "why": "w"},
    ]
    bootstrap_from_seeds("topic", seeds, topic_type="mixed")
    md = (tmp_path / "plan.md").read_text()
    assert "**Core vocabulary:**" not in md


def test_plan_sections_merges_core_vocabulary_into_header():
    """utils._plan_sections must pull the vocab-line into HEADER so the gate sees it."""
    from lra.utils import extract_topic_keywords_tiered
    plan_md = (
        "# Plan: generic topic title\n"
        "\n"
        "**Core vocabulary:** jamming, ELINT, pulse deinterleaving\n"
        "\n"
        "[FOCUS] ...\n"
        "\n"
        "## [TODO]\n"
        "- [T1] task one\n"
    )
    header_kws, _ = extract_topic_keywords_tiered(plan_md)
    # vocab terms made it into header (and thus header_kws)
    assert "jamming" in header_kws
    assert "elint" in header_kws
    assert "deinterleaving" in header_kws


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
    # reset static plan first (as reset_research does)
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
    """LLM returns junk → retry also junk → static-vocab fallback from query.

    After adding retry+static fallback (P2) the function returns True if
    at least one domain term (≥4 chars) was extracted from the query. Plan stays
    static, but core_vocabulary is filled with keywords from the query itself.
    """
    from lra import pipeline
    from lra import plan as plan_mod
    plan_mod.reset("retrieval augmented generation pipelines")  # multi-word query
    static_titles = {t.title for t in plan_mod.load().tasks}

    monkeypatch.setattr(pipeline, "_run_agent", _fake_run_agent("lol not json"))
    monkeypatch.setattr(pipeline, "build_bot", lambda *a, **kw: object())

    ok = pipeline._bootstrap_initial_plan("retrieval augmented generation pipelines")
    assert ok is True  # static-vocab fallback applied
    loaded = plan_mod.load()
    # Static tasks untouched
    assert {t.title for t in loaded.tasks} == static_titles
    # But vocabulary is now filled from the query
    assert len(loaded.core_vocabulary) >= 2
    assert "retrieval" in loaded.core_vocabulary or "generation" in loaded.core_vocabulary


def test_pipeline_bootstrap_survives_llm_exception(_isolated_all, monkeypatch):
    """LLM raises exception → retry also raises → static-vocab fallback triggers."""
    from lra import pipeline
    from lra import plan as plan_mod
    plan_mod.reset("electronic warfare jamming detection")

    def _boom(*a, **kw):
        raise RuntimeError("mlx crashed")
    monkeypatch.setattr(pipeline, "build_bot", _boom)

    ok = pipeline._bootstrap_initial_plan("electronic warfare jamming detection")
    assert ok is True  # static fallback
    loaded = plan_mod.load()
    assert loaded is not None
    assert len(loaded.core_vocabulary) >= 2


def test_pipeline_bootstrap_no_fallback_for_empty_query(_isolated_all, monkeypatch):
    """If the query is too short to extract vocab (1-3 chars), fallback is empty → False."""
    from lra import pipeline
    from lra import plan as plan_mod
    plan_mod.reset("xx")

    monkeypatch.setattr(pipeline, "_run_agent", _fake_run_agent("garbage"))
    monkeypatch.setattr(pipeline, "build_bot", lambda *a, **kw: object())

    ok = pipeline._bootstrap_initial_plan("xx")
    assert ok is False
    assert plan_mod.load() is not None  # static plan survived


def test_cfg_flag_default_on():
    """CFG['dynamic_initial_plan'] defaults to True (guard in research_loop)."""
    from lra.config import CFG
    assert CFG.get("dynamic_initial_plan", True) is True
