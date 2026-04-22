"""Domain gate в AppendNotes: блокирует paper из смежного домена (citation laundering source).

Реальный кейс: запрос про EW/ELINT → explorer принёс ComVo (audio vocoder).
Gate должен заблокировать по недостаточному overlap с topic keywords из plan.md.
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
    "## [TODO]\n"
    "- [T1] Spectrum sensing algorithms for cognitive EW in contested environments\n"
    "- [T2] Software-defined radar (SDR) for agile electronic support measures\n"
    "- [T4] ELINT signal fingerprinting and geolocation precision limits\n"
)


def test_extract_topic_keywords_drops_generic():
    from lra.utils import extract_topic_keywords
    kws = extract_topic_keywords(_EW_PLAN)
    # Доменные термины — есть
    assert "electronic" in kws
    assert "warfare" in kws
    assert "elint" in kws
    assert "spectrum" in kws
    assert "radar" in kws
    # Generic — отфильтрованы
    assert "modern" not in kws
    assert "approach" not in kws
    assert "approaches" not in kws


def test_domain_gate_blocks_comvo_style_laundering(_isolated):
    """Реальный кейс: explorer принёс ComVo (audio vocoder) по запросу про EW."""
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    # Имитируем успешный поиск: ComVo лежит в kb (id известен), но paper про аудио.
    kb.add(kb.Atom(
        id="2603.11589", kind="paper", topic="audio neural vocoder",
        title="ComVo complex-valued neural vocoder",
        claim=("complex-valued neural vocoder for waveform generation from mel-spectrogram, "
               "adversarial training, Korea University, phase representation iSTFT"),
    ))
    result = tools.AppendNotes().call(
        {"content": "[2603.11589] adversarial training для EW waveform generation"}
    )
    assert "ОТКАЗ" in result
    assert "domain gate" in result
    # notes.md не создан, rejected.jsonl записан
    assert not (_isolated / "notes.md").exists()
    rej = (_isolated / "rejected.jsonl").read_text(encoding="utf-8")
    assert "2603.11589" in rej


def test_domain_gate_passes_relevant_paper(_isolated):
    from lra import kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    # Real EW paper: overlap с topic kws = {electronic, warfare, spectrum, radar, elint}
    kb.add(kb.Atom(
        id="2401.00001", kind="paper", topic="electronic warfare",
        title="Cognitive electronic warfare spectrum sensing",
        claim=("cognitive electronic warfare system with radar signal classification, "
               "ELINT fingerprinting in contested spectrum environments"),
    ))
    result = tools.AppendNotes().call(
        {"content": "[2401.00001] cognitive EW spectrum sensing"}
    )
    assert "ОТКАЗ" not in result
    assert (_isolated / "notes.md").exists()


def test_domain_gate_bypassed_for_reflection_without_ids(_isolated):
    """Reflection-заметки без arxiv-id не фильтруются domain gate."""
    from lra import tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    result = tools.AppendNotes().call(
        {"content": "## Lesson: hf_papers 'generic AI' даёт нерелевантные результаты"}
    )
    assert "ОТКАЗ" not in result
    assert (_isolated / "notes.md").exists()


def test_domain_gate_bypassed_without_plan(_isolated):
    """Если plan.md не создан — gate пропускает (тема ещё не задана)."""
    from lra import kb, tools
    kb.add(kb.Atom(id="2401.00001", kind="paper", topic="t", claim="c"))
    # plan.md нет
    result = tools.AppendNotes().call({"content": "[2401.00001] early finding"})
    assert "ОТКАЗ" not in result


def test_domain_gate_lenient_flag(_isolated, monkeypatch):
    """CFG['strict_domain_gate']=False → gate отключён."""
    from lra import config, kb, tools
    (_isolated / "plan.md").write_text(_EW_PLAN, encoding="utf-8")
    kb.add(kb.Atom(
        id="2603.11589", kind="paper", topic="audio",
        title="audio vocoder", claim="mel-spectrogram waveform generation",
    ))
    original_get = config.CFG.get
    monkeypatch.setattr(
        config.CFG, "get",
        lambda k, d=None: False if k == "strict_domain_gate" else original_get(k, d),
    )
    result = tools.AppendNotes().call(
        {"content": "[2603.11589] off-topic paper"}
    )
    assert "ОТКАЗ" not in result
