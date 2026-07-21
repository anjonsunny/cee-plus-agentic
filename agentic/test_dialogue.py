"""Hermetic tests for the dialogue agent (LangGraph) and its tools.

No model, no network: the LLM is a scripted function; records are tmp
fixtures. Covered: tool lookups (including honest errors), the graph's
tool loop and its cap, per-turn provenance traces, and thread memory
across turns.

Run:  pytest agentic/test_dialogue.py -q
"""
from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("langgraph")

from agentic import agent_tools  # noqa: E402
from agentic.dialogue import MAX_TOOL_ROUNDS, respond  # noqa: E402

# ── Fixture: a fake run record on disk ──────────────────────────────────

RECORD = {
    "image_path": "/x/B_pool.jpg", "image_size": [100, 80], "caption": "cap",
    "entity_source": "vlm",
    "detected_objects": [
        {"object_id": "child_1", "label": "child", "family": "person",
         "state": "drowning", "state_kind": "at_risk",
         "description": "struggling in water", "bbox": [5, 5, 50, 50],
         "box_source": "dino_matched", "box_confidence": 0.77,
         "anchor_bbox": [4, 4, 52, 52], "mask_path": None, "label_note": "",
         "vocab_extension": False, "family_name_as_label": False}],
    "unlocalized": [], "notes": [],
    "repair_trace": {"rounds": [], "clean_on_arrival": True,
                     "stopped_reason": "clean"},
}


@pytest.fixture()
def fake_records(tmp_path, monkeypatch):
    pdir = tmp_path / "perception"
    pdir.mkdir()
    (pdir / "B_pool__perception.json").write_text(json.dumps(RECORD))
    monkeypatch.setattr(agent_tools, "PERCEPTION_DIR", pdir)
    monkeypatch.setattr(agent_tools, "UI_RUNS_DIR", tmp_path / "none")


# ── Tools ───────────────────────────────────────────────────────────────


def test_list_and_summary(fake_records):
    assert agent_tools.list_runs() == {"runs": ["B_pool"]}
    s = agent_tools.get_run_summary("B_pool")
    assert s["at_risk"] == ["child_1"] and s["repair"]["stopped_reason"] == "clean"


def test_entity_lookup_and_honest_miss(fake_records):
    e = agent_tools.get_entity("B_pool", "child_1")
    assert e["state"] == "drowning" and e["box_source"] == "dino_matched"
    miss = agent_tools.get_entity("B_pool", "shark_1")
    assert "error" in miss and miss["known"] == ["child_1"]
    assert "error" in agent_tools.get_run_summary("Z_nope")


def test_call_tool_never_raises():
    assert "error" in agent_tools.call_tool("no_such_tool", {})
    assert "error" in agent_tools.call_tool("get_entity", {"bogus": 1})


# ── The graph ───────────────────────────────────────────────────────────


def _tc(name: str, args: dict) -> dict:
    return {"id": f"call_{name}", "function": {
        "name": name, "arguments": json.dumps(args)}}


def scripted_llm(script):
    """An llm_fn that plays back a list of assistant messages."""
    state = {"i": 0}

    def fn(messages, tools):
        msg = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        return msg

    return fn


def test_tool_loop_answers_with_provenance(fake_records):
    llm = scripted_llm([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("get_entity", {"run": "B_pool", "object_id": "child_1"})]},
        {"role": "assistant",
         "content": "child_1 is drowning (at risk), box from the detector."},
    ])
    out = respond(f"t-{uuid.uuid4().hex}", "what about child_1?", llm_fn=llm)
    assert "drowning" in out["answer"]
    assert out["tool_trace"][0]["tool"] == "get_entity"
    assert out["tool_trace"][0]["result"]["state"] == "drowning"


def test_thread_memory_accumulates(fake_records):
    tid = f"t-{uuid.uuid4().hex}"
    llm1 = scripted_llm([{"role": "assistant", "content": "Hello, ask me about runs."}])
    respond(tid, "hi", llm_fn=llm1)

    seen = {}

    def llm2(messages, tools):
        seen["n_user"] = sum(1 for m in messages if m.get("role") == "user")
        return {"role": "assistant", "content": "second answer"}

    out = respond(tid, "and again", llm_fn=llm2)
    assert seen["n_user"] == 2                 # first turn remembered
    assert out["answer"] == "second answer"


def test_tool_round_cap_ends_the_loop(fake_records):
    """An agent that keeps calling tools forever is stopped by the cap."""
    endless = {"role": "assistant", "content": None,
               "tool_calls": [_tc("list_runs", {})]}
    out = respond(f"t-{uuid.uuid4().hex}", "loop!", llm_fn=scripted_llm([endless]))
    assert len(out["tool_trace"]) <= MAX_TOOL_ROUNDS
    assert out["answer"] == "(no answer produced)"


def test_focus_run_lands_in_system_prompt(fake_records):
    seen = {}

    def llm(messages, tools):
        seen["system"] = messages[0]["content"]
        return {"role": "assistant", "content": "ok"}

    respond(f"t-{uuid.uuid4().hex}", "hi", llm_fn=llm, focus_run="B_pool")
    assert "B_pool" in seen["system"]
