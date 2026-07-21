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
"""
from __future__ import annotations

import json
import operator
import os
import sys
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
_GRAPH_CACHE: dict[int, Any] = {}


def build_graph(llm_fn: LlmFn | None = None, focus_run: str | None = None):
    """Assemble and compile the two-node tool loop."""
    llm_fn = llm_fn or _ollama_llm
    focus_line = (f"The user is currently looking at run '{focus_run}'; "
                  f"when they do not name a run, they mean this one."
                  if focus_run else "")
    system = {"role": "system",
              "content": SYSTEM_PROMPT.format(focus_line=focus_line)}

    def llm_node(state: AgentState) -> dict[str, Any]:
        answer = llm_fn([system] + state["messages"], TOOL_SCHEMAS)
        return {"messages": [answer]}

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        out_msgs: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for tc in last.get("tool_calls") or []:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = call_tool(name, args)
            trace.append({"tool": name, "args": args, "result": result})
            out_msgs.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "content": json.dumps(result)})
        return {"messages": out_msgs, "tool_trace": trace}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if m.get("role") == "tool")
        if last.get("tool_calls") and rounds < MAX_TOOL_ROUNDS:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile(checkpointer=_CHECKPOINTER)


# ── Public entry point ──────────────────────────────────────────────────


def respond(thread_id: str, user_text: str,
            llm_fn: LlmFn | None = None,
            focus_run: str | None = None) -> dict[str, Any]:
    """One conversational turn on a thread. Returns the assistant's answer,
    this turn's tool trace (the answer's provenance), and the full thread
    transcript for rendering."""
    graph = build_graph(llm_fn=llm_fn, focus_run=focus_run)
    config = {"configurable": {"thread_id": thread_id}}
    before = graph.get_state(config)
    trace_before = len((before.values or {}).get("tool_trace", []))

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_text}], "tool_trace": []},
        config=config,
    )
    answer = ""
    for m in reversed(result["messages"]):
        if m.get("role") == "assistant" and m.get("content"):
            answer = m["content"]
            break
    return {
        "answer": answer or "(no answer produced)",
        "tool_trace": result["tool_trace"][trace_before:],
        "messages": result["messages"],
    }
