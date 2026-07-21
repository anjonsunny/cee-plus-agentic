"""Dialogue agent tools: structured lookup over the run records.

Design (AGENTIC_PLAN Stage 17): the agent is READ-ONLY and terminal. It
queries records and talks to the user; its output never re-enters the
pipeline and never reaches the subject VLM. Run data is addressable by
key, so these are lookups, not retrieval: *concepts* get retrieved (the
rulebook), *data* gets queried (these functions).

Every tool returns plain JSON-serializable data. The agent's faithfulness
discipline (dialogue.py) requires each factual claim to trace to one of
these results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENES_DIR = REPO_ROOT / "experiments" / "agentic_scenes"
PERCEPTION_DIR = SCENES_DIR / "perception"
UI_RUNS_DIR = REPO_ROOT / "exports" / "agentic_runs"


# ── Run resolution ──────────────────────────────────────────────────────


def _record_paths() -> dict[str, Path]:
    """All known run records by name: the six frozen worked-example scenes
    plus every saved UI run."""
    out: dict[str, Path] = {}
    if PERCEPTION_DIR.exists():
        for p in sorted(PERCEPTION_DIR.glob("*__perception.json")):
            out[p.name.replace("__perception.json", "")] = p
    if UI_RUNS_DIR.exists():
        for p in sorted(UI_RUNS_DIR.glob("*/")):
            recs = list(p.glob("*__perception.json"))
            if recs:
                out[p.name] = recs[0]
    return out


def _load(run: str) -> dict[str, Any] | None:
    path = _record_paths().get(run)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


# ── The tools ───────────────────────────────────────────────────────────


def list_runs() -> dict[str, Any]:
    """Every run the agent can talk about."""
    return {"runs": sorted(_record_paths().keys())}


def get_run_summary(run: str) -> dict[str, Any]:
    """Caption, entity roster, repair verdict, and box provenance counts
    for one run."""
    rec = _load(run)
    if rec is None:
        return {"error": f"no record named '{run}'", "known": sorted(_record_paths())}
    objs = rec.get("detected_objects", [])
    trace = rec.get("repair_trace") or {}
    return {
        "run": run,
        "caption": rec.get("caption", ""),
        "entities": [{"object_id": o["object_id"], "state": o["state"],
                      "state_kind": o["state_kind"]} for o in objs],
        "hazards": [o["object_id"] for o in objs
                    if o["state_kind"] == "hazard_bearing"],
        "at_risk": [o["object_id"] for o in objs if o["state_kind"] == "at_risk"],
        "unlocalized": rec.get("unlocalized", []),
        "repair": {"stopped_reason": trace.get("stopped_reason"),
                   "rounds": len(trace.get("rounds", [])),
                   "clean_on_arrival": trace.get("clean_on_arrival", False)},
    }


def get_entity(run: str, object_id: str) -> dict[str, Any]:
    """Everything recorded about one entity: label, state, description,
    geometry provenance (anchor vs final box, source, confidence), flags."""
    rec = _load(run)
    if rec is None:
        return {"error": f"no record named '{run}'"}
    for o in rec.get("detected_objects", []):
        if o["object_id"] == object_id:
            return {k: o.get(k) for k in
                    ("object_id", "label", "family", "state", "state_kind",
                     "description", "bbox", "anchor_bbox", "box_source",
                     "box_confidence", "label_note", "vocab_extension",
                     "family_name_as_label", "mask_path")}
    return {"error": f"no entity '{object_id}' in '{run}'",
            "known": [o["object_id"] for o in rec.get("detected_objects", [])]}


def get_repair_story(run: str) -> dict[str, Any]:
    """The Loop 1 history: every round, every violation with its
    instruction, what changed, and how the loop ended."""
    rec = _load(run)
    if rec is None:
        return {"error": f"no record named '{run}'"}
    trace = rec.get("repair_trace")
    if not trace:
        return {"run": run, "repair": "loop not run for this record"}
    return {
        "run": run,
        "clean_on_arrival": trace.get("clean_on_arrival", False),
        "stopped_reason": trace.get("stopped_reason"),
        "rounds": [{
            "round": r.get("round_number"),
            "changed": r.get("changed"),
            "violations": [{"kind": v.get("kind"),
                            "raw_label": v.get("raw_label"),
                            "instruction": v.get("instruction", "")[:300]}
                           for v in r.get("violations", [])],
        } for r in trace.get("rounds", [])],
    }


# ── OpenAI-style schemas + dispatch (what the LLM sees) ─────────────────

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_runs",
        "description": "List every analyzed run/scene the assistant can discuss.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_run_summary",
        "description": "Summary of one run: caption, entities with states, "
                       "hazards, at-risk entities, repair verdict.",
        "parameters": {"type": "object", "properties": {
            "run": {"type": "string", "description": "run name, e.g. B_pool"}},
            "required": ["run"]}}},
    {"type": "function", "function": {
        "name": "get_entity",
        "description": "Full record of one entity in a run, including box "
                       "provenance (anchor vs final, source, confidence).",
        "parameters": {"type": "object", "properties": {
            "run": {"type": "string"},
            "object_id": {"type": "string", "description": "e.g. child_1"}},
            "required": ["run", "object_id"]}}},
    {"type": "function", "function": {
        "name": "get_repair_story",
        "description": "The repair loop history of a run: rounds, violations, "
                       "instructions sent, and how it ended.",
        "parameters": {"type": "object", "properties": {
            "run": {"type": "string"}}, "required": ["run"]}}},
]

_DISPATCH = {
    "list_runs": list_runs,
    "get_run_summary": get_run_summary,
    "get_entity": get_entity,
    "get_repair_story": get_repair_story,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call. Unknown tools and bad arguments return an
    error object rather than raising: the agent must SAY a lookup failed,
    never invent an answer around a crash."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
