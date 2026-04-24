"""Domain gate in AppendNotes + hf_papers kb auto-save.

Real failure modes from logs:
1. ComVo [2603.11589] (audio vocoder) — single 'intelligence' overlap, passed the old gate.
2. Emotional-support [2508.12935] — single 'electronic' overlap, passed the old gate.
3. Both landed in kb.jsonl via auto-save in hf_papers BEFORE AppendNotes could cut them.

Tiered gate: requires ≥1 core hit (from HEADER of plan.md) AND ≥2 total overlap.
"""
import pytest


@pytest.fixture
def _isolated(tmp_path, monkeypatch):
    from lra import config, kb, memory, tools
    monkeypatch.setattr(config, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(config, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(config, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(config, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(kb, "KB_PATH", tmp_path / "kb.jsonl")
    monkeypatch.setattr(memory, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(tools, "NOTES_PATH", tmp_path / "notes.md")
    monkeypatch.setattr(tools, "PLAN_PATH", tmp_path / "plan.md")
    monkeypatch.setattr(tools, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    return tmp_path


_EW_PLAN = (
    "# Plan: Modern approaches in electronic warfare (EW) and electronic intelligence "
    "(ELINT) technologies in the military\n\n"
    "**Core vocabulary:** ELINT, SIGINT, ESM, ECM, jammers, receivers, spectrum sensing, "
    "direction finding, pulse deinterleaving, anti-jamming, cognitive radar, LPI/LPD\n\n"
    "## [TODO]\n"
    "- [T1] Spectrum sensing algorithms for cognitive EW in contested environments\n"
    "- [T2] Software-defined radar (SDR) for agile electronic support measures\n"
    "- [T4] ELINT signal fingerprinting and geolocation precision limits\n"
)


def test_tiered_keywords_separate_header_from_seeds():
    from lra.utils import extract_topic_keywords_tiered
    header, seeds = extract_topic_keywords_tiered(_EW_PLAN)
    assert {"electronic", "warfare", "elint", "intelligence"}.issubset(header)
    # Core vocabulary terms are now in the header too (vocab-line is part of header).
    assert "jammers" in header or "receivers" in header
    # Seeds are disjoint from the header, contain domain specifics not in vocab
    assert "fingerprinting" in seeds or "geolocation" in seeds
    # Generic noise is filtered on both tiers
    for noise in ("modern", "approach", "contested", "environments", "canyons",
                  "between", "challenges", "advanced", "novel", "algorithms"):
        assert noise not in header, f"{noise} leaked into header"
        assert noise not in seeds, f"{noise} leaked into seeds"


def test_domain_gate_blocks_comvo_no_core_hit(_isolated):
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    kb.add(kb.Atom(
        id="2603.11589", kind="paper", topic="audio neural vocoder",
        title="ComVo complex-valued neural vocoder",
        claim=("complex-valued neural vocoder for waveform generation from mel-spectrogram, "
               "adversarial training, Korea University, phase representation iSTFT"),
    ))
    result = tools.AppendNotes().call(
        {"content": "[2603.11589] adversarial training for EW waveform generation"}
    )
    assert "REJECTED" in result and "no_core_hit" in result
    assert not (_isolated / "notes.md").exists()
    import json
    rej_line = (_isolated / "rejected.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    rej = json.loads(rej_line)
    assert rej["reason"] == "no_core_hit"
    assert "2603.11589" in rej["ids"]


def test_domain_gate_blocks_single_generic_overlap(_isolated):
    """emotional-support-conversations with a single hit on 'intelligence'.

    The old gate (threshold 2 on a flat set) could let it through: intelligence + electronic
    both in plan.md. Tiered: core hit exists but no seeds hit → needs an extra
    seed term to pass, or ≥2 core-hit.
    """
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    kb.add(kb.Atom(
        id="2508.12935", kind="paper", topic="dialog",
        title="Emotional support dialogue via RL",
        claim=("reinforcement learning for emotional support dialogue generation, "
               "human feedback and empathy"),
    ))
    result = tools.AppendNotes().call({"content": "[2508.12935] intelligence of dialog agents"})
    assert "REJECTED" in result


def test_domain_gate_passes_relevant_paper(_isolated):
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    kb.add(kb.Atom(
        id="2401.00001", kind="paper", topic="electronic warfare",
        title="Cognitive electronic warfare spectrum sensing for ELINT",
        claim=("cognitive electronic warfare system with radar signal classification, "
               "ELINT fingerprinting in contested spectrum environments"),
    ))
    result = tools.AppendNotes().call({"content": "[2401.00001] cognitive EW spectrum sensing"})
    assert "REJECTED" not in result
    assert (_isolated / "notes.md").exists()


def test_domain_gate_bypassed_for_reflection_without_ids(_isolated):
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    result = tools.AppendNotes().call(
        {"content": "## Lesson: hf_papers 'generic AI' gives irrelevant results"}
    )
    assert "REJECTED" not in result


def test_domain_gate_bypassed_without_plan(_isolated):
    from lra import kb, tools
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t", claim="c"))
    result = tools.AppendNotes().call({"content": "[2401.00001] early finding"})
    assert "REJECTED" not in result


def test_domain_gate_lenient_flag(_isolated, monkeypatch):
    from lra import config, kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    kb.add(kb.Atom(id="2603.11589", kind="paper", topic="audio",
                   title="audio vocoder", claim="mel-spectrogram waveform generation"))
    original_get = config.CFG.get
    monkeypatch.setattr(
        config.CFG, "get",
        lambda k, d=None: False if k == "strict_domain_gate" else original_get(k, d),
    )
    result = tools.AppendNotes().call({"content": "[2603.11589] off-topic paper"})
    assert "REJECTED" not in result


def test_gate_paper_for_kb_blocks_comvo(_isolated):
    """hf_papers kb auto-save gate: off-topic paper must not land in kb.jsonl."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2603.11589",
        "ComVo complex-valued neural vocoder",
        "complex-valued neural vocoder for waveform generation from mel-spectrogram "
        "with adversarial training and phase representation",
    )
    assert passed is False
    assert reason.startswith("anti_keyword")  # vocoder is in ANTI_KEYWORDS, reason=anti_keyword:vocoder(_isolated):
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2512.05753",
        "FARDA: Fast Anti-Jamming Radar Deployment via Deep Reinforcement Learning",
        "end-to-end DRL for electronic warfare radar deployment, jamming resistance, "
        "spectrum sensing in contested environments, ELINT scenarios",
    )
    assert passed is True
    assert reason == "passed"


def test_gate_paper_for_kb_bypass_without_plan(_isolated):
    from lra import tools
    passed, reason, _o, _h = tools.gate_paper_for_kb("2401.00001", "any", "any content")
    assert passed is True


_NARROW_PLAN = (
    "# Plan: electronic warfare and ELINT\n\n"
    "**Core vocabulary:** ELINT, ECM, jammers\n"
)  # header = {electronic, warfare, elint, intelligence, jammers, ecm} ≈ 4-6 words


def test_gate_paper_for_kb_adaptive_threshold_narrow_header(_isolated):
    """Narrow header (≤4 core-kws) → 1 hit is enough, paper is not cut as "weak_overlap"."""
    from lra import tools
    (_isolated / "plan.md").write_text(_NARROW_PLAN, encoding="utf-8")
    # cognitive radar jamming — 1 hit (no direct 'electronic'/'warfare' but title mentions)
    passed, reason, o_h, h = tools.gate_paper_for_kb(
        "2501.11111",
        "Cognitive radar jamming detection via deep learning",
        "electronic countermeasures in contested spectrum using CNN classifier",
    )
    assert passed is True
    assert reason == "passed"


def test_gate_paper_for_kb_logs_rejection_to_jsonl(_isolated):
    """Skip in gate_paper_for_kb must be written to rejected.jsonl with reason kb_autosave:*."""
    import json

    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason, o_h, h = tools.gate_paper_for_kb(
        "2603.11589",
        "ComVo vocoder",
        "mel-spectrogram waveform generation",
    )
    assert passed is False
    tools._log_kb_rejected("2603.11589", "ComVo vocoder", reason, o_h, h, source="hf_papers")
    lines = (_isolated / "rejected.jsonl").read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["reason"].startswith("kb_autosave:anti_keyword")  # anti_keyword:vocoder
    assert entry["paper_id"] == "2603.11589"
    assert entry["source"] == "hf_papers"


def test_anti_keyword_blocks_jailbreak_paper(_isolated):
    """An LLM-safety/jailbreak paper must not pass the domain gate."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2507.22564",
        "Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs",
        "We exploit cognitive biases to jailbreak safety alignment in large language models.",
    )
    assert passed is False
    assert "anti_keyword" in reason  # e.g. anti_keyword:jailbreak or anti_keyword:safety alignment


def test_anti_keyword_does_not_block_relevant_ew_paper(_isolated):
    """A relevant EW paper without anti-keywords passes."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2506.11048",
        "CMuSeNet: Complex-Valued Multi-Signal Segmentation",
        "Electronic warfare spectrum sensing using complex-valued neural networks for ELINT.",
    )
    assert passed is True


def test_gate_repo_for_kb_blocks_comvo(_isolated):
    """The GitHub gate must block an audio vocoder repo."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason = tools.gate_repo_for_kb(
        "hs-oh-prml/ComVo",
        "neural vocoder for audio synthesis from mel-spectrogram",
    )
    assert passed is False
    assert "vocoder" in reason or "anti_keyword" in reason


def test_gate_repo_for_kb_passes_ew_repo(_isolated):
    """An EW-relevant repo must pass."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    passed, reason = tools.gate_repo_for_kb(
        "example/radar-jamming-drl",
        "Deep reinforcement learning for electronic warfare anti-jamming radar spectrum sensing",
    )
    assert passed is True


# ── Fail-closed when bootstrap planner fails (no **Core vocabulary:** in plan.md) ──
_PLAN_NO_VOCAB = (
    "# Plan: electronic warfare and ELINT\n\n"
    "## [TODO]\n"
    "- [T1] some seed\n"
)


def test_gate_paper_fails_closed_without_vocabulary(_isolated):
    """Bootstrap failed → no Core vocabulary → even a relevant paper DOES NOT pass.

    Previously the gate worked on 4 header words → let mines through (cognitive radar
    in minesweeping context) as if EW. Now fail-closed: until the user
    explicitly allows CFG['allow_no_vocab']=True, everything is blocked.
    """
    from lra import tools
    (_isolated / "plan.md").write_text(_PLAN_NO_VOCAB, encoding="utf-8")
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2401.00001",
        "Cognitive electronic warfare spectrum sensing",
        "ELINT signal classification in contested EW environments",
    )
    assert passed is False
    assert reason == "no_vocabulary"


def test_gate_repo_fails_closed_without_vocabulary(_isolated):
    from lra import tools
    (_isolated / "plan.md").write_text(_PLAN_NO_VOCAB, encoding="utf-8")
    passed, reason = tools.gate_repo_for_kb(
        "example/ew-radar",
        "EW radar spectrum sensing for ELINT",
    )
    assert passed is False
    assert reason == "no_vocabulary"


def test_append_notes_fails_closed_without_vocabulary(_isolated):
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_PLAN_NO_VOCAB, encoding="utf-8")
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="ew",
                   title="EW radar", claim="electronic warfare spectrum sensing"))
    result = tools.AppendNotes().call({"content": "[2401.00001] EW radar paper"})
    assert "REJECTED" in result and "no_vocabulary" in result


def test_allow_no_vocab_escape_hatch(_isolated, monkeypatch):
    """CFG['allow_no_vocab']=True allows operation without vocabulary (at your own risk)."""
    from lra import config, tools
    (_isolated / "plan.md").write_text(_PLAN_NO_VOCAB, encoding="utf-8")
    monkeypatch.setitem(config.CFG.extra, "allow_no_vocab", True)
    passed, reason, _o, _h = tools.gate_paper_for_kb(
        "2401.00001",
        "Cognitive electronic warfare spectrum sensing",
        "ELINT signal classification in contested EW environments",
    )
    assert passed is True
    assert reason == "passed"
