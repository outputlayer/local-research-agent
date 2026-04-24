"""Tests for the citation validator — fully without hf CLI (run_hf_info=False)."""
from lra.validator import validate_draft_ids


def test_no_ids_returns_zeros():
    assert validate_draft_ids("just text", "just notes", run_hf_info=False) == (0, [], [])


def test_id_missing_from_notes_marked_invalid():
    draft = "According to [2301.12345], it works."
    notes = "# Notes\n(no id)"
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
    # id is in notes, but the context around it in the draft is about something else
    notes = (
        "[2301.12345] Chinchilla scaling laws compute tokens parameters Hoffmann\n"
    )
    draft = (
        "Quantum gravity string theory gauge supersymmetry [2301.12345]"
    )
    valid, invalid, suspicious = validate_draft_ids(draft, notes, run_hf_info=False)
    assert valid == 1  # id is formally present in notes
    assert invalid == []
    assert len(suspicious) == 1
    assert "2301.12345" in suspicious[0]
    assert "overlap=" in suspicious[0]


def test_citation_laundering_caught_at_threshold_5():
    # Real scenario from a live run: the writer takes an arxiv-id from notes
    # but re-attributes the claim to the query topic. The jargon partially
    # overlaps (complex-valued/adversarial/neural), but the topic differs:
    # notes → audio vocoder, draft → EW countermeasures.
    # Threshold 3 let this through; 5 catches it.
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
    assert len(suspicious) == 1, f"citation laundering must be caught, overlap={suspicious}"


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
