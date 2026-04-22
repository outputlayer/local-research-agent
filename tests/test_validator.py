"""Тесты валидатора цитат — полностью без hf CLI (run_hf_info=False)."""
from lra.validator import validate_draft_ids


def test_no_ids_returns_zeros():
    assert validate_draft_ids("just text", "just notes", run_hf_info=False) == (0, [], [])


def test_id_missing_from_notes_marked_invalid():
    draft = "Согласно [2301.12345], это работает."
    notes = "# Notes\n(нет id)"
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert invalid == ["2301.12345"]
    assert valid == 0


def test_id_with_good_citation_passes():
    notes = (
        "# Notes\n\n"
        "## Scaling laws\n"
        "[2301.12345] Chinchilla optimal training compute budget tokens parameters ratio\n"
        "Authors: Hoffmann. Key fact: compute-optimal ratio is 20 tokens per parameter.\n"
    )
    draft = (
        "# Report\n"
        "Scaling laws [2301.12345] describe compute-optimal training ratios for parameters and tokens."
    )
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert valid == 1
    assert invalid == []
    assert suspicious == []


def test_suspicious_when_citation_context_mismatches_notes():
    # id есть в notes, но в draft'е контекст вокруг него совсем про другое
    notes = (
        "[2301.12345] Chinchilla scaling laws compute tokens parameters Hoffmann\n"
    )
    draft = (
        "Квантовая гравитация струнная теория калибровочные суперсимметрия [2301.12345]"
    )
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert valid == 1  # id формально существует в notes
    assert invalid == []
    assert len(suspicious) == 1
    assert "2301.12345" in suspicious[0]
    assert "overlap=" in suspicious[0]


def test_citation_laundering_caught_at_threshold_5():
    # Реальный сценарий из live run'а: writer берёт arxiv-id из notes,
    # но переатрибутирует claim под тему запроса. Жаргон частично
    # пересекается (complex-valued/adversarial/neural), но тема разная:
    # notes → audio vocoder, draft → EW countermeasures.
    # Threshold 3 такое пропускал, 5 ловит.
    notes = (
        "[2603.11589] ComVo audio vocoder complex-valued adversarial training "
        "waveform generation Korea University phase representation iSTFT"
    )
    draft = (
        "Cognitive EW countermeasures [2603.11589] applies complex-valued adversarial "
        "jamming against enemy radar systems with 72% success rate"
    )
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert valid == 1
    assert len(suspicious) == 1, f"citation laundering должен ловиться, overlap={suspicious}"


def test_multiple_ids_mixed():
    notes = (
        "[2301.11111] Transformer architecture attention multihead normalization training\n"
        "[2302.22222] Mixture experts routing sparse conditional computation scaling\n"
    )
    draft = (
        "The transformer [2301.11111] uses attention and multihead normalization training.\n"
        "Sparse mixture [2302.22222] employs routing for conditional computation scaling.\n"
        "And a hallucinated [2312.99999] reference.\n"
    )
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert invalid == ["2312.99999"]
    assert valid == 2
    assert suspicious == []
