"""Dialogue agent v0, built on LangGraph from the start (Sunny's call).

THE GRAPH
=========
Two nodes and one loop, the canonical tool-calling shape:

        START -> llm --(tool_calls?)--> tools -> llm -> ... -> END

  - `llm`   asks the dialogue model (Ollama, OpenAI-compatible endpoint)
            what to do next, given the conversation and the tool schemas.
  - `tools` executes each requested tool (agent_tools.py) and appends the
            results as tool messages.
The loop runs until the model answers in plain text, or the tool-round
cap trips (an agent must not be able to loop forever).

STATE AND MEMORY
================
Conversation state is LangGraph state with a checkpointer: each thread
(one per UI session/run) accumulates its messages, so "what about the
other child?" resolves against earlier turns. This is conversation
memory, not content memory: run data stays in the records and is queried
fresh on every question (a cached copy would go stale).

FAITHFULNESS DISCIPLINE
=======================
The system prompt binds the agent to its tools: every factual claim must
come from a tool result; when a lookup returns nothing, SAY SO. The
tool_trace in the state records every call and result per turn, so each
answer's provenance is inspectable in the UI: this is the same
"evidence, then verdict" ethos as the rest of CEE+.

MODEL
=====
DIALOGUE_MODEL env var, default qwen2.5:7b (a TEXT model: deliberately
not the subject VLM, per the plan's instrument/subject separation; full
training-family diversity matters for pathology judging later, not for
lookups). Tests inject a scripted llm_fn and never touch a model.

KNOWN LIMITS (deliberate prototype scope, reviewed 2026-07-21)
==============================================================
- MemorySaver is in-process: conversation history dies with the app,
  is not shared across processes, and grows unboundedly (no pruning or
  summarization yet). Swap in a SQLite checkpointer when that matters.
- thread_id comes from the UI run, not the browser/user: two people on
  a shared server viewing one run would share a conversation. Fine for
  a single-user local prototype.
- Record immutability is a POLICY, not an enforcement: frozen scenes
  and completed run exports are written once and never edited, so tool
  results cannot go stale within a session. If records ever become
  mutable, version them by run id/hash.
- The tool_trace is a RETRIEVAL trace (what was looked up), not a proof
  of answer faithfulness. The unverified-id check in respond() catches
  entity ids with no evidence; a full claim-vs-evidence validator is
  Stage 23's job.
"""
from __future__ import annotations

import json
import operator
import os
import re
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from agentic.agent_tools import TOOL_SCHEMAS, call_tool  # noqa: E402

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the CEE+ analysis assistant. You answer questions
about analyzed emergency scenes using ONLY your tools.

Rules you must follow:
- Every factual claim must come from a tool result in this conversation.
  If you have not looked it up, look it up before answering.
- If a lookup returns an error or nothing, say exactly that. Never guess,
  never fill gaps from general knowledge. An invented answer is the exact
  failure this project exists to expose.
- Refer to entities by their object_id (child_1, tanker_truck_1).
- Phrase findings as what the PERCEPTION RECORD reports, never as ground
  truth about the scene: say "the record identifies child_1 as drowning",
  not "child_1 is drowning". The records are model outputs under
  evaluation; this project must never launder them into facts.
- Be concise and concrete. One or two short paragraphs at most.
{focus_line}"""


class AgentState(TypedDict):
    # Full OpenAI-format message dicts; operator.add appends across turns
    # so the checkpointer accumulates the conversation per thread.
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_trace: Annotated[list[dict[str, Any]], operator.add]


# ── The Ollama call (injectable for tests) ──────────────────────────────


def _ollama_llm(messages: list[dict[str, Any]],
                tools: list[dict[str, Any]]) -> dict[str, Any]:
    """One chat completion against the local dialogue model. Returns the
    assistant message dict (may carry tool_calls)."""
    import requests

    api_url = os.getenv("QWEN_API_URL", "http://localhost:11434/v1/chat/completions")
    r = requests.post(api_url, json={
        "model": os.getenv("DIALOGUE_MODEL", "qwen2.5:7b"),
        "messages": messages,
        "tools": tools,
        "temperature": 0,
    }, timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


LlmFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]

# One shared checkpointer so every compiled graph variant (live or test
# llm_fn) sees the same per-thread conversation memory.
_CHECKPOINTER = MemorySaver()

# One lock per thread_id: two questions submitted quickly on the same
# conversation must SERIALIZE, or they race the checkpointer (both load
# the same state, one overwrites the other, trace_before slices wrong).
# The guarantee lives here with the agent, not with whichever UI calls it.
_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(thread_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(thread_id, threading.Lock())


# Sentinel key marking a synthetic "tool budget exhausted" denial message
# (see cap_node); also used to count denial batches when routing.
_CAP_KEY = "cap_denied"

StepFn = Callable[[dict[str, Any]], None]


def _rounds_this_turn(messages: list[dict[str, Any]]) -> int:
    """Count tool-result messages since the LAST user message.

    The cap must be PER USER TURN. Counting all tool messages in the
    thread (the original bug) meant the budget was spent across the
    conversation's lifetime: after ~3 normal questions the agent silently
    lost tool access and degraded into exactly the ungrounded guesser the
    system prompt forbids."""
    n = 0
    for m in reversed(messages):
        if m.get("role") == "user":
            break
        if m.get("role") == "tool":
            n += 1
    return n


def _denial_batches_this_turn(messages: list[dict[str, Any]]) -> int:
    """How many times THIS turn the model was already told the budget is
    gone (a batch = one denied assistant tool request)."""
    batches, in_batch = 0, False
    for m in reversed(messages):
        if m.get("role") == "user":
            break
        is_denial = m.get("role") == "tool" and _CAP_KEY in (m.get("content") or "")
        if is_denial and not in_batch:
            batches += 1
        in_batch = is_denial
    return batches


# Entity ids look like child_1, tanker_truck_1, fire_1.
_ID_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")


def _ids_in(obj: Any) -> set[str]:
    """All entity-id-shaped strings anywhere in a JSON-ish value."""
    if isinstance(obj, str):
        return set(_ID_RE.findall(obj))
    if isinstance(obj, dict):
        return _ids_in(list(obj.keys())) | _ids_in(list(obj.values()))
    if isinstance(obj, (list, tuple)):
        out: set[str] = set()
        for v in obj:
            out |= _ids_in(v)
        return out
    return set()


def unverified_ids(answer: str, trace: list[dict[str, Any]]) -> list[str]:
    """Entity ids the answer mentions that appear in NO tool result this
    turn. Deterministic and cheap: this flags (it does not block) answers
    whose named entities have no retrieved evidence behind them. It is
    NOT a faithfulness proof (see KNOWN LIMITS)."""
    evidence: set[str] = set()
    for t in trace:
        evidence |= _ids_in(t.get("result"))
    return sorted(_ids_in(answer) - evidence)


def build_graph(llm_fn: LlmFn | None = None, focus_run: str | None = None,
                on_step: StepFn | None = None):
    """Assemble and compile the two-node tool loop.

    `on_step` (optional) receives the agent's trajectory as it happens:
      {"step": "thinking"}                       before each model call
      {"step": "tool_call", tool, args}          when the model picks a tool
      {"step": "tool_result", tool, ok, summary} when the tool returns
      {"step": "answer", "text": ...}            the final reply
    The UI renders this live, so tool use is visible before the answer:
    the same show-the-work ethos as the pipeline's event stream."""
    llm_fn = llm_fn or _ollama_llm
    emit = on_step or (lambda _e: None)
    focus_line = (f"The user is currently looking at run '{focus_run}'; "
                  f"when they do not name a run, they mean this one."
                  if focus_run else "")
    system = {"role": "system",
              "content": SYSTEM_PROMPT.format(focus_line=focus_line)}

    def llm_node(state: AgentState) -> dict[str, Any]:
        emit({"step": "thinking"})
        answer = llm_fn([system] + state["messages"], TOOL_SCHEMAS)
        if answer.get("content") and not answer.get("tool_calls"):
            emit({"step": "answer", "text": answer["content"]})
        return {"messages": [answer]}

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        out_msgs: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for tc in last.get("tool_calls") or []:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            try:
                args, parse_error = json.loads(raw), None
            except json.JSONDecodeError as exc:
                # A malformed tool call is EVIDENCE of agent failure, not
                # noise: preserve the raw string and the parse error in
                # the trace instead of silently degrading to {} (which
                # would masquerade as an ordinary missing-argument miss).
                args, parse_error = {}, str(exc)
            emit({"step": "tool_call", "tool": name, "args": args,
                  **({"parse_error": parse_error} if parse_error else {})})
            if parse_error:
                result = {"error": f"malformed tool arguments: {parse_error}",
                          "raw_arguments": raw}
            else:
                result = call_tool(name, args)
            ok = "error" not in result
            summary = (result.get("error") if not ok
                       else ", ".join(list(result)[:4]))
            emit({"step": "tool_result", "tool": name, "ok": ok,
                  "summary": str(summary)})
            entry = {"tool": name, "args": args, "result": result}
            if parse_error:
                entry.update(raw_arguments=raw, parse_error=parse_error)
            trace.append(entry)
            out_msgs.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "content": json.dumps(result)})
        return {"messages": out_msgs, "tool_trace": trace}

    def cap_node(state: AgentState) -> dict[str, Any]:
        """The model asked for a tool after the per-turn budget was spent.
        We must still answer each tool_call id (a dangling tool request is
        a malformed OpenAI sequence that poisons the NEXT turn), so append
        synthetic denial results and record the denial in the trace."""
        last = state["messages"][-1]
        out_msgs, trace = [], []
        denial = {"error": ("tool budget for this turn is exhausted; "
                            "answer from evidence already gathered"),
                  _CAP_KEY: True}
        for tc in last.get("tool_calls") or []:
            name = tc["function"]["name"]
            emit({"step": "tool_result", "tool": name, "ok": False,
                  "summary": "denied: tool budget reached"})
            trace.append({"tool": name, "args": {}, "result": denial})
            out_msgs.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "content": json.dumps(denial)})
        return {"messages": out_msgs, "tool_trace": trace}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if not last.get("tool_calls"):
            return END                     # plain answer -> done
        if _rounds_this_turn(state["messages"]) < MAX_TOOL_ROUNDS:
            return "tools"                 # budget left -> execute
        return "cap"                       # budget spent -> deny, validly

    def route_after_cap(state: AgentState) -> str:
        # One denial earns the model a final plain-text chance; if it
        # keeps demanding tools after TWO denials, end the turn (the
        # sequence is valid: every tool_call got a denial response).
        if _denial_batches_this_turn(state["messages"]) >= 2:
            return END
        return "llm"

    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)
    g.add_node("cap", cap_node)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route,
                            {"tools": "tools", "cap": "cap", END: END})
    g.add_edge("tools", "llm")
    g.add_conditional_edges("cap", route_after_cap, {"llm": "llm", END: END})
    return g.compile(checkpointer=_CHECKPOINTER)


# ── Public entry point ──────────────────────────────────────────────────


def respond(thread_id: str, user_text: str,
            llm_fn: LlmFn | None = None,
            focus_run: str | None = None,
            on_step: StepFn | None = None) -> dict[str, Any]:
    """One conversational turn on a thread. Returns the assistant's answer,
    this turn's tool trace (the answer's retrieval provenance), the ids the
    answer mentions WITHOUT evidence (unverified_ids: the UI badges these,
    it never blocks), and the full thread transcript for rendering.
    `on_step` streams the trajectory live. Turns on the same thread_id
    serialize on a lock (see _THREAD_LOCKS)."""
    graph = build_graph(llm_fn=llm_fn, focus_run=focus_run, on_step=on_step)
    config = {"configurable": {"thread_id": thread_id}}
    with _lock_for(thread_id):
        before = graph.get_state(config)
        trace_before = len((before.values or {}).get("tool_trace", []))

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_text}],
             "tool_trace": []},
            config=config,
        )
    answer = ""
    for m in reversed(result["messages"]):
        if m.get("role") == "assistant" and m.get("content"):
            answer = m["content"]
            break
    turn_trace = result["tool_trace"][trace_before:]
    return {
        "answer": answer or "(no answer produced)",
        "tool_trace": turn_trace,
        "unverified_ids": unverified_ids(answer, turn_trace),
        "messages": result["messages"],
    }
