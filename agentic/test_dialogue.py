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
    executed = [t for t in out["tool_trace"]
                if "cap_denied" not in t.get("result", {})]
    assert len(executed) <= MAX_TOOL_ROUNDS
    assert out["answer"] == "(no answer produced)"


def test_cap_resets_per_user_turn(fake_records):
    """Codex finding #1: the budget must be PER TURN. Several normal
    turns (1 tool call each) on one thread must never exhaust it —
    the old counter summed tool messages across the whole conversation
    and silently cut off tool access around turn MAX_TOOL_ROUNDS."""
    tid = f"t-{uuid.uuid4().hex}"
    for i in range(MAX_TOOL_ROUNDS + 3):       # more turns than the cap
        llm = scripted_llm([
            {"role": "assistant", "content": None,
             "tool_calls": [_tc("get_entity",
                                {"run": "B_pool", "object_id": "child_1"})]},
            {"role": "assistant", "content": f"turn {i}: child_1 drowning."},
        ])
        out = respond(tid, f"question {i}", llm_fn=llm)
        # the tool must have EXECUTED this turn, not been denied
        assert out["tool_trace"], f"turn {i}: tool call was not executed"
        assert out["tool_trace"][0]["result"].get("state") == "drowning"


def test_cap_stop_leaves_no_dangling_tool_calls(fake_records):
    """When the cap trips, every assistant tool_call must still get a
    tool response (else the stored sequence is malformed OpenAI and
    poisons the NEXT turn), and the next turn must work normally."""
    tid = f"t-{uuid.uuid4().hex}"
    endless = {"role": "assistant", "content": None,
               "tool_calls": [_tc("list_runs", {})]}
    out = respond(tid, "loop!", llm_fn=scripted_llm([endless]))
    msgs = out["messages"]
    answered = {m["tool_call_id"] for m in msgs if m.get("role") == "tool"}
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            assert tc["id"] in answered, f"dangling tool_call {tc['id']}"
    # the poisoned-next-turn regression: a normal turn still succeeds
    llm = scripted_llm([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("get_entity", {"run": "B_pool", "object_id": "child_1"})]},
        {"role": "assistant", "content": "the record identifies child_1 as drowning."}])
    nxt = respond(tid, "and child_1?", llm_fn=llm)
    assert "drowning" in nxt["answer"]
    assert nxt["tool_trace"][0]["result"]["state"] == "drowning"


def test_concurrent_turns_on_one_thread_serialize(fake_records):
    """Codex finding #5: two rapid submissions on one thread_id race the
    checkpointer. respond() now serializes them on a per-thread lock:
    both complete, both user turns land in the final transcript."""
    import threading as _th
    tid = f"t-{uuid.uuid4().hex}"
    results: list[dict] = []

    def slow_llm(messages, tools):
        import time
        time.sleep(0.05)                       # widen the race window
        return {"role": "assistant", "content": f"seen {len(messages)} msgs"}

    def go(q):
        results.append(respond(tid, q, llm_fn=slow_llm))

    threads = [_th.Thread(target=go, args=(f"q{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 2 and all(r["answer"] for r in results)
    final = max(results, key=lambda r: len(r["messages"]))["messages"]
    users = [m["content"] for m in final if m.get("role") == "user"]
    assert sorted(users) == ["q0", "q1"]       # neither turn was lost


def test_malformed_tool_arguments_preserved_as_evidence(fake_records):
    """Codex finding #10: bad JSON args are agent-failure EVIDENCE. The
    trace must keep the raw string and parse error, not silently {}."""
    bad_call = {"id": "call_bad", "function": {
        "name": "get_entity", "arguments": '{"run": "B_pool", broken'}}
    llm = scripted_llm([
        {"role": "assistant", "content": None, "tool_calls": [bad_call]},
        {"role": "assistant", "content": "that lookup failed."},
    ])
    steps: list[dict] = []
    out = respond(f"t-{uuid.uuid4().hex}", "?", llm_fn=llm, on_step=steps.append)
    entry = out["tool_trace"][0]
    assert entry["raw_arguments"].startswith('{"run"')
    assert entry["parse_error"]
    assert "malformed" in entry["result"]["error"]
    call_step = next(s for s in steps if s["step"] == "tool_call")
    assert call_step.get("parse_error")        # visible in the UI trajectory
    result_step = next(s for s in steps if s["step"] == "tool_result")
    assert not result_step["ok"]


def test_unverified_ids_flag_evidence_free_mentions(fake_records):
    """Codex findings #2/#9, our show-don't-block middle path: an id in
    the answer with no tool evidence gets flagged; evidenced ids don't."""
    llm = scripted_llm([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("get_entity", {"run": "B_pool", "object_id": "child_1"})]},
        {"role": "assistant",
         "content": "child_1 is drowning and shark_9 is circling."},
    ])
    out = respond(f"t-{uuid.uuid4().hex}", "?", llm_fn=llm)
    assert out["unverified_ids"] == ["shark_9"]

    ungrounded = scripted_llm([
        {"role": "assistant", "content": "child_1 is fine."}])  # no tools at all
    out2 = respond(f"t-{uuid.uuid4().hex}", "?", llm_fn=ungrounded)
    assert out2["unverified_ids"] == ["child_1"]


def test_system_prompt_demands_record_speak(fake_records):
    """Codex finding #8: answers must attribute claims to the perception
    record, never state them as ground truth."""
    seen = {}

    def llm(messages, tools):
        seen["system"] = messages[0]["content"]
        return {"role": "assistant", "content": "ok"}

    respond(f"t-{uuid.uuid4().hex}", "hi", llm_fn=llm)
    assert "record identifies child_1 as drowning" in seen["system"]
    assert "never launder" in seen["system"]


def test_trajectory_steps_stream_in_order(fake_records):
    """on_step receives the agent's path: thinking -> tool_call ->
    tool_result -> thinking -> answer (Sunny: show the steps before the
    answer, so tool use is visible)."""
    llm = scripted_llm([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("get_entity", {"run": "B_pool", "object_id": "child_1"})]},
        {"role": "assistant", "content": "child_1 is drowning."},
    ])
    steps: list[dict] = []
    respond(f"t-{uuid.uuid4().hex}", "child_1?", llm_fn=llm,
            on_step=steps.append)
    kinds = [s["step"] for s in steps]
    assert kinds == ["thinking", "tool_call", "tool_result", "thinking", "answer"]
    assert steps[1]["tool"] == "get_entity"
    assert steps[2]["ok"] and "object_id" in steps[2]["summary"]
    assert steps[4]["text"] == "child_1 is drowning."


def test_trajectory_marks_failed_lookup(fake_records):
    llm = scripted_llm([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("get_entity", {"run": "B_pool", "object_id": "shark_1"})]},
        {"role": "assistant", "content": "no such entity."},
    ])
    steps: list[dict] = []
    respond(f"t-{uuid.uuid4().hex}", "shark?", llm_fn=llm, on_step=steps.append)
    result_step = next(s for s in steps if s["step"] == "tool_result")
    assert not result_step["ok"] and "no entity" in result_step["summary"]


def test_transcript_renders_live_trajectory():
    """The UI transcript shows steps above the answer, pulsing while
    pending."""
    from agentic.ui import agent_transcript_component
    log = [{"q": "why?", "pending": True, "a": None, "steps": [
        {"step": "thinking"},
        {"step": "tool_call", "tool": "get_run_summary", "args": {"run": "B_pool"}},
    ]}]
    out = str(agent_transcript_component(log))
    assert "get_run_summary" in out and "tl-row" in out and "now" in out
    log[0].update(pending=False, a="because child_1 is drowning",
                  steps=log[0]["steps"] + [
                      {"step": "tool_result", "tool": "get_run_summary",
                       "ok": True, "summary": "run, caption"},
                      {"step": "answer", "text": "because child_1 is drowning"}])
    done = str(agent_transcript_component(log))
    assert "because child_1 is drowning" in done and "now" not in done


def test_transcript_badges_unverified_ids():
    """An answer naming entities with no retrieved evidence gets a
    visible badge under the bubble (flag, never block)."""
    from agentic.ui import agent_transcript_component
    log = [{"q": "?", "pending": False, "a": "shark_9 is circling",
            "steps": [], "unverified": ["shark_9"]}]
    out = str(agent_transcript_component(log))
    assert "not in retrieved evidence" in out and "shark_9" in out
    clean = [{"q": "?", "pending": False, "a": "all good",
              "steps": [], "unverified": []}]
    assert "not in retrieved evidence" not in str(agent_transcript_component(clean))


def test_focus_run_lands_in_system_prompt(fake_records):
    seen = {}

    def llm(messages, tools):
        seen["system"] = messages[0]["content"]
        return {"role": "assistant", "content": "ok"}

    respond(f"t-{uuid.uuid4().hex}", "hi", llm_fn=llm, focus_run="B_pool")
    assert "B_pool" in seen["system"]
