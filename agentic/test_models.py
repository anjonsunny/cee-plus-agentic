

# ── the JSON constraint is a per-subject seat setting (F_park 2026-09-10) ──

def test_json_mode_is_on_for_qwen25vl_and_off_for_thinking_models(monkeypatch):
    """F50 dropped response_format for qwen3-vl (thinking: content came back
    '{}'); qwen2.5vl NEEDS it — without it the park scene looped on
    'bbox_2, bbox_3, ...' to the 16k wall, three runs in a row."""
    from agentic import models
    monkeypatch.delenv("SUBJECT_JSON_MODE", raising=False)
    assert models.subject_json_mode("qwen2.5vl:7b") is True
    assert models.subject_format_kwargs("qwen2.5vl:7b") == {
        "response_format": {"type": "json_object"}}
    assert models.subject_json_mode("qwen3-vl:8b") is False
    assert models.subject_format_kwargs("qwen3-vl:8b") == {}
    assert models.subject_json_mode("gemma4:26b") is True


def test_json_mode_env_override_wins(monkeypatch):
    from agentic import models
    monkeypatch.setenv("SUBJECT_JSON_MODE", "0")
    assert models.subject_json_mode("qwen2.5vl:7b") is False
    monkeypatch.setenv("SUBJECT_JSON_MODE", "1")
    assert models.subject_json_mode("qwen3-vl:8b") is True


def test_every_subject_call_site_uses_the_seat():
    """No subject payload hard-codes the constraint either way."""
    import pathlib
    for f in ("assessment.py", "recommend.py", "perception.py"):
        src = pathlib.Path(__file__).with_name(f).read_text()
        assert "subject_format_kwargs()" in src, f
        assert '"response_format"' not in src, f
