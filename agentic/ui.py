"""Agentic CEE+ live UI (v0): watch the perception pipeline work.

WHAT THIS IS
============
A separate Dash app (port 8060; main.py's app is untouched). Upload an
image, type a caption, hit ANALYZE: the Stage 1 agentic pipeline runs in a
background thread and the screen follows it live:

  - STAGE RAIL   the pipeline as stations (Perceive -> Repair -> Ground ->
                 Bind -> Mask -> Assemble); the active station pulses; the
                 Repair station is a visible loop with round pips.
  - TICKETS      each rule violation is a card: amber while open, folded
                 under a green FIXED stamp, or pinned under a gray STOOD
                 ITS GROUND seal. Expanding a ticket shows the exact
                 instruction the Rulebook sent and what the Model changed.
  - INSTRUMENTS  live counters: entities, violations open/resolved, repair
                 round pips, bind quality, elapsed per stage.
  - SCENE VIEW   the image; rough VLM anchors appear as faint dashed
                 outlines, final boxes snap in colored by state kind
                 (red hazard / orange at-risk / blue normal), solid =
                 detector-bound, dashed = SAM fallback. Open violations pin
                 amber chips to their entity. Clicking a box opens the
                 entity inspector (full provenance chain).
  - SCRUBBER     a slider over the event stream; drag left to rewind the
                 whole screen (rail, tickets, scene) to any moment; at the
                 right edge it follows the live run.

HOW IT WORKS
============
The pipeline emits structured events via run_perception(on_event=...)
(Stage 23 observability in embryo). Events append to an in-memory list per
run. Every UI element is a PURE FUNCTION of events[:k]: live mode renders
k = len(events); scrubbing renders a smaller k. Replay mode builds a
synthetic event stream from any saved perception JSON (the six frozen
worked-example scenes appear in the dropdown), so the app is fully usable
without Ollama.

Run:  python -m agentic.ui        then open http://localhost:8060
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import dash
from dash import ALL, Input, Output, State, ctx, dcc, html

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preload shared heavy libraries in the MAIN thread, before any background
# pipeline thread exists. Dash's JSON encoder touches pandas.NaT while
# serializing responses; if a worker thread is importing the ML stack
# (which pulls pandas in) at that same moment, the request sees a
# partially initialized module ("pandas has no attribute 'NaT'", live run
# 2026-07-21). Importing here serializes it once and forever.
for _mod in ("pandas", "numpy", "PIL.Image"):
    try:
        __import__(_mod)
    except ImportError:
        pass

# F46. Two families of number on this screen run in OPPOSITE directions, and
# both were labelled "score", which left the reader to work out which way was
# good. One word each, and the definition in a tooltip so it costs no space.
#
#   "clean"   1.00 = nothing broken.  HIGHER IS BETTER.
#   "unsure"  0.00 = it said the same thing every time. HIGHER IS WORSE.
#
# Neither is a pass fraction — both are 1 - x/(x + size) shapes — so nothing
# on screen may present them as "N of M checks", which would be a denominator
# we do not have.
_CLEAN_TIP = ("1.00 = nothing broken. Falls as rule breaks get more severe. "
              "Not a pass fraction — there is no 'N out of M' here.")
_UNSURE_TIP = ("0.00 = it gave the same answer every time. Rises as the "
               "answers disagree with each other across re-asks.")

SCENES_DIR = REPO_ROOT / "experiments" / "agentic_scenes"
PERCEPTION_DIR = SCENES_DIR / "perception"
ASSESSMENT_DIR = SCENES_DIR / "assessment"
STAGES = ["Perceive", "Repair", "Ground", "Bind", "Mask", "Assemble"]

# Severity buckets tint the verdict ribbon and the PHASE 1 card accents.
BUCKET_CSS = {"none": "#16a34a", "minor": "#eab308",
              "serious": "#f97316", "catastrophic": "#ef4444"}

STATE_KIND_CSS = {
    "hazard_bearing": "#ef4444",
    "at_risk": "#f97316",
    "normal": "#3b82f6",
    "unknown": "#94a3b8",
}

STAGE_COLORS = {
    "Perceive": "#3b82f6", "Repair": "#f59e0b", "Ground": "#8b5cf6",
    "Bind": "#06b6d4", "Mask": "#ec4899", "Assemble": "#16a34a",
}


def _timeline_glyph(text: str) -> tuple[str, str]:
    """(glyph, css modifier) for an activity line, by what it reports."""
    if text.startswith("found:"):
        return "!", "warn"
    if "asking the model" in text or text.endswith("..."):
        return "…", "ask"
    if ("done:" in text or "resolved" in text or "matched" in text
            or "captured" in text or "saved" in text or "found" in text
            or "entities" in text or "fallback" in text):
        return "✓", "ok"
    if "stood its ground" in text or "cap reached" in text:
        return "■", "stood"
    return "·", "info"

# In-memory run registry: {run_id: {"events": [...], "image_src": str,
# "error": str|None, "done": bool}}. One user, local tool: a dict is enough.
RUNS: dict[str, dict[str, Any]] = {}


# ── Event-stream derivation (pure; the heart of live + scrub unity) ─────


def _restore_epoch0_perception(d: dict) -> None:
    """A petition that adds nothing leaves the run on the ORIGINAL record —
    so the picture must show that record.

    `petition_started` clears the live accumulators so the re-perception can
    build its own panels. Nothing restored them when the second look was
    REFUSED, so the overlay kept drawing the rejected pass: boxes for entities
    the two-witness rule turned away, and none for the entities every
    downstream stage actually reasoned about.

    The rejected pass is not erased — it stays in the petition panel as the
    evidence of what was refused and why. One rule holds: the picture always
    shows the record the run reasoned on."""
    e0 = d.get("epoch0") or {}
    for key in ("bound", "anchors", "result", "perceive_entities",
                "repair_entities_latest", "violations", "rounds_used",
                "stop_reason", "stages", "stage_activities"):
        if key in e0:
            d[key] = e0[key]


def derive(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold an event prefix into the render state. Pure function: same
    events in, same screen out; the scrubber is just a shorter prefix."""
    d: dict[str, Any] = {
        "image_size": None, "caption": "", "image_name": "",
        "stages": {s: {"status": "pending", "seconds": None, "info": ""} for s in STAGES},
        "violations": {},          # key -> {..., status: open|fixed|stood}
        "rounds_used": 0, "stop_reason": None,
        "anchors": [],             # entities with anchor boxes (dashed phase)
        "bound": {},               # object_id -> {bbox, box_source, confidence}
        "result": None,            # final PerceptionResult dict
        "repair_entities_latest": [],  # raw entity list during repair rounds
        # The image narrates the run: `activity` is the latest thing
        # happening, rendered as a ribbon ON the scene, with an optional
        # spotlight target (the entity currently being worked on).
        "activity": {"text": "", "oid": None, "busy": False},
        "perceive_entities": [],   # the model's FIRST answer (pre-repair)
        # Per-station activity feed (the Cowork-style narration Sunny asked
        # for): each stage accumulates high-level lines; while the stage is
        # active its newest line renders as "happening now".
        "stage_activities": {s: [] for s in STAGES},
        # True between repair_round_started and its round_done: the open
        # violations are literally being fixed right now.
        "round_in_progress": False,
        # ── Stage 2 (PHASE 1 · ASSESS) ──────────────────────────────────
        # status pending|active|done; verdict = the SceneAssessment dict;
        # uncertainty = the assess_uncertainty event payload (channel 2);
        # probes/notes/violations accumulate for the activity feed.
        "assess": {"status": "pending", "verdict": None, "uncertainty": None,
                   "context": {}, "probes": [], "notes": [],
                   "violations": [], "activities": [],
                   # Stage-1-style lifecycle tickets for assessment
                   # violations: kind|evidence -> {kind, evidence, status:
                   # open|fixing|fixed|stood}
                   "tickets": {},
                   # every assess_verdict event, in order: [0] = canonical
                   # (pre-reflection), [-1] = final. The ledger diffs them.
                   "verdict_history": []},
        # RAG shadow rows (rag/both retrieval mode): exact-key vs RAG top-1
        # for the rules each stage actually quoted. Stage 1 (repair) and
        # Stage 2 (assess/reflection) each get their own panel.
        "retrieval_shadow": None,           # Stage 2 (assess)
        "retrieval_shadow_perceive": None,  # Stage 1 (repair)
        # why the RAG lookup used word-count instead of the real embedding
        # (None when the real vector path ran). Same stack both stages.
        "retrieval_backend_reason": None,
        # Auto result diff (both/rag mode, only when a fired rule differs):
        # whether RAG's rule choice actually moved the final answer.
        "retrieval_result_diff": None,
        # Stage 4 (recommendation, Phase 1a): the full Stage4Result dump, or
        # an error string. None until the run reaches Stage 4.
        "stage4": None,
        "stage4_error": None,
        # Stage 4 progress, for the card's running/done badge.
        "stage4_started": False,
        "stage4_marks": set(),
        "stage4_probe": 0,          # live count of uncertainty re-asks so far
    }
    for ev in events:
        t = ev.get("type")
        if t == "run_started":
            d["image_size"] = ev.get("image_size")
            d["caption"] = ev.get("caption", "")
            d["image_name"] = ev.get("image_name", "")
        elif t == "stage_started":
            s = ev.get("stage")
            if s in d["stages"]:
                d["stages"][s]["status"] = "active"
        elif t == "stage_done":
            s = ev.get("stage")
            if s in d["stages"]:
                d["stages"][s]["status"] = "done"
                d["stages"][s]["seconds"] = ev.get("seconds")
                extras = {k: v for k, v in ev.items()
                          if k not in ("type", "stage", "seconds", "entities", "t")}
                d["stages"][s]["info"] = ", ".join(f"{k}={v}" for k, v in extras.items())
                if s == "Perceive":
                    d["perceive_entities"] = ev.get("entities", [])
        elif t == "violation_found":
            key = f"{ev.get('kind')}|{ev.get('raw_label')}|{ev.get('entity_index')}"
            d["violations"].setdefault(key, {
                "kind": ev.get("kind"), "raw_label": ev.get("raw_label"),
                "entity_index": ev.get("entity_index"),
                "instruction": ev.get("instruction", ""),
                "first_round": ev.get("round", 1), "status": "open",
            })
        elif t == "repair_round_done":
            d["rounds_used"] = max(d["rounds_used"], int(ev.get("round", 0)))
            d["repair_entities_latest"] = ev.get("entities", [])
        elif t == "repair_stopped":
            d["stop_reason"] = ev.get("reason")
            remaining = {
                f"{v.get('kind')}|{v.get('raw_label')}|{v.get('entity_index')}"
                for v in ev.get("remaining", [])
            }
            for key, v in d["violations"].items():
                v["status"] = "stood" if key in remaining else "fixed"
        elif t == "anchors_ready":
            d["anchors"] = ev.get("entities", [])
        elif t == "entity_bound":
            d["bound"][ev.get("object_id")] = {
                "bbox": ev.get("bbox"), "box_source": ev.get("box_source"),
                "confidence": ev.get("confidence", 0.0),
            }
        elif t == "assembled":
            d["result"] = ev.get("result")
        elif t == "run_error":
            d["error"] = ev.get("message", "unknown error")
            # F31: a run that DIED mid-stage used to leave that stage sitting
            # at "active" forever, so a run which errored 10ms in — Ollama not
            # listening — rendered as an eternal spinner on Perceive. The
            # failure was recorded in the event stream and shown nowhere.
            # Whichever stage was in flight is marked failed, and carries the
            # message, so the screen says what happened instead of implying
            # work is still going on.
            for name, st in d["stages"].items():
                if st.get("status") == "active":
                    st["status"] = "failed"
                    st["info"] = str(d["error"])[:160]
                    d["stage_activities"].setdefault(name, []).append(
                        f"failed — {str(d['error'])[:160]}")
            d["activity"] = {"text": f"run failed — {str(d['error'])[:120]}",
                             "oid": None, "busy": False}

        # Stage 4 starts when the recommend node begins — show the badge
        # active immediately (the recommend call is the longest sub-step).
        if t == "stage_started" and ev.get("stage") == "recommend":
            d["stage4_started"] = True

        # ── Stage 2 events (stage name "assess" is not in the Phase 0
        # STAGES list, so the generic stage handlers above ignore it). ──
        A = d["assess"]
        if t == "stage_started" and ev.get("stage") == "assess":
            A["status"] = "active"
            A["activities"].append("judging the scene from declared state "
                                   "(text-only, no image)...")
            d["activity"] = {"text": "assessing the scene from state...",
                             "oid": None, "busy": True}
        elif t == "duplicate_merged":
            line = (f"same individual counted twice: {ev.get('dropped')} "
                    f"merged into {ev.get('kept')} (IoU {ev.get('iou')})")
            d["stage_activities"]["Assemble"].append(line)
            d["activity"] = {"text": line, "oid": ev.get("kept"),
                             "busy": True}
            # The dropped copy leaves the PICTURE too (Sunny: person_2/3
            # boxes stayed on the image after the merge — the scene draws
            # from the bind-step accumulators, which predate the merge).
            dropped = ev.get("dropped")
            d["bound"].pop(dropped, None)
            d["anchors"] = [a for a in d["anchors"]
                            if a.get("object_id") != dropped]
        elif t == "retrieval_shadow":
            # stage "perceive" -> Stage 1 panel; "assess" (or unset, for old
            # replays) -> Stage 2 panel.
            if ev.get("stage") == "perceive":
                d["retrieval_shadow_perceive"] = ev.get("rows", [])
            else:
                d["retrieval_shadow"] = ev.get("rows", [])
            if ev.get("backend_reason"):
                d["retrieval_backend_reason"] = ev.get("backend_reason")
        elif t == "retrieval_result_diff":
            d["retrieval_result_diff"] = {
                "changed": ev.get("changed"),
                "rulebook_line": ev.get("rulebook_line"),
                "rag_line": ev.get("rag_line"),
                "error": ev.get("error"),
            }
        elif t == "stage4_result":
            d["stage4"] = ev.get("result")
        elif t == "stage4_error":
            d["stage4_error"] = ev.get("message")
        elif t == "recommendations_ready":
            d["stage4_started"] = True
            d["stage4_marks"].add("recommend")
        elif t == "graph_a_built":
            d["stage4_marks"].add("graph_a")
        elif t == "graph_b_built":
            d["stage4_marks"].add("graph_b")
        elif t == "targets_picked":
            d["stage4_marks"].add("picks")
        elif t == "recommend_probe":
            d["stage4_probe"] = d.get("stage4_probe", 0) + 1
        elif t == "recommend_uncertainty_ready":
            d["stage4_marks"].add("uncertainty")
        elif t == "card_judge_ready":
            d["stage4_marks"].add("card_judge")
        elif t == "graph_judge_ready":
            d["stage4_marks"].add("graph_judge")
        elif t == "runoff_judged":
            d["stage4_runoff"] = d.get("stage4_runoff", 0) + 1
        elif t == "trust_ready":
            # runoff has no fixed application count (0-2 fire), so its step is
            # marked done when trust starts — trust follows it immediately.
            d["stage4_marks"].add("runoff")
            d["stage4_marks"].add("trust")
        elif t == "hazard_derived":
            line = (f"{ev.get('medium')} '{ev.get('was')}' → "
                    f"'{ev.get('now')}' — {ev.get('victim')} is "
                    f"{ev.get('victim_state')} (medium-bound rule, "
                    f"derived in code)")
            A.setdefault("derived_hazards", []).append(line)
            A["activities"].append(f"⚑ derived hazard: {line}")
        elif t == "assess_context":
            A["context"] = {"n_entities": ev.get("n_entities"),
                            "hazard_ids": ev.get("hazard_ids", []),
                            "at_risk_ids": ev.get("at_risk_ids", []),
                            "n_spatial_hints": ev.get("n_spatial_hints"),
                            "spatial_pairs": ev.get("spatial_pairs", [])}
            hz = ", ".join(ev.get("hazard_ids", []) or []) or "none"
            ar = ", ".join(ev.get("at_risk_ids", []) or []) or "none"
            A["activities"].append(f"evidence in: hazards [{hz}] · "
                                   f"at-risk [{ar}]")
        elif t == "assess_parse_note":
            A["notes"].append(ev.get("note", ""))
            A["activities"].append(f"coerced: {ev.get('note', '')}")
        elif t == "assess_violation":
            A["violations"].append({"kind": ev.get("kind"),
                                    "evidence": ev.get("evidence", "")})
            key = f"{ev.get('kind')}|{ev.get('evidence', '')[:60]}"
            A["tickets"].setdefault(key, {"kind": ev.get("kind"),
                                          "evidence": ev.get("evidence", ""),
                                          "status": "open"})
            A["activities"].append(f"found: {str(ev.get('kind', '')).replace('_', ' ')}")
            d["activity"] = {"text": f"verdict violation: "
                                     f"{str(ev.get('kind', '')).replace('_', ' ')}",
                             "oid": None, "busy": True}
        elif t == "assess_verdict":
            A["verdict"] = {k: ev.get(k) for k in
                            ("scenario", "disaster_type", "level", "bucket",
                             "self_confidence", "n_violations",
                             "threats", "at_risk")}
            A["verdict_history"].append(dict(A["verdict"]))
            A["activities"].append(
                f"verdict: {ev.get('scenario')} · {ev.get('disaster_type')} "
                f"· level {ev.get('level')} ({ev.get('bucket')})")
            d["activity"] = {"text": f"verdict: {ev.get('scenario')} · "
                                     f"{ev.get('disaster_type')} · "
                                     f"level {ev.get('level')}",
                             "oid": None, "busy": False}
        elif t == "assess_probe":
            A["probes"].append({k: ev.get(k) for k in
                                ("index", "scenario", "level", "bucket",
                                 "threat_ids", "at_risk_ids")})
            A["activities"].append(f"probe {int(ev.get('index', 0)) + 1}: "
                                   f"{ev.get('scenario')} · level "
                                   f"{ev.get('level')} ({ev.get('bucket')})")
            d["activity"] = {"text": f"probing verdict stability "
                                     f"({int(ev.get('index', 0)) + 1})...",
                             "oid": None, "busy": True}
        elif t == "petition_started" and ev.get("target") == "stage2":
            import copy as _copy
            # Stage-2 petition: the image and the record are untouched,
            # so the perception panels stay live. Freeze only the
            # assessment as epoch 0 and let the fresh answer build below.
            d["epoch0"] = {"assess": _copy.deepcopy(d["assess"])}
            A = d["assess"] = {"status": "active", "verdict": None,
                               "uncertainty": None, "context": {},
                               "probes": [], "notes": [],
                               "violations": [], "activities": [],
                               "tickets": {}, "verdict_history": []}
            A["petition"] = {"status": "in_flight", "target": "stage2",
                             "reasons": ev.get("reasons", []),
                             "added": [], "removed": [], "outcome": None}
            why = ", ".join(str(r.get("kind", "?")).replace("_", " ")
                            for r in ev.get("reasons", []))
            A["activities"].append(f"PETITION within Stage 2: asking the "
                                   f"question again, fresh — unresolved: "
                                   f"{why}")
            d["activity"] = {"text": "re-asking the assessment question "
                                     "(fresh, one try)...",
                             "oid": None, "busy": True}
        elif t == "petition_started":
            import copy as _copy
            # Freeze the ENTIRE first pass as epoch 0 — both stages —
            # then reset the live accumulators so the re-run builds its
            # own panels underneath. Nothing is overwritten: the before/
            # after comparison is the petition's whole evidentiary value.
            d["epoch0"] = _copy.deepcopy({
                "stages": d["stages"], "violations": d["violations"],
                "rounds_used": d["rounds_used"],
                "stop_reason": d["stop_reason"],
                "perceive_entities": d["perceive_entities"],
                "stage_activities": d["stage_activities"],
                "anchors": d["anchors"], "bound": d["bound"],
                "result": d["result"],
                "repair_entities_latest": d["repair_entities_latest"],
                "round_in_progress": False,
                "activity": {"text": "", "oid": None, "busy": False},
                "assess": d["assess"],
            })
            d["stages"] = {st: {"status": "pending", "seconds": None,
                                "info": ""} for st in STAGES}
            d["violations"] = {}
            d["rounds_used"] = 0
            d["stop_reason"] = None
            d["perceive_entities"] = []
            d["stage_activities"] = {st: [] for st in STAGES}
            d["anchors"] = []
            d["bound"] = {}
            d["result"] = None
            d["repair_entities_latest"] = []
            A = d["assess"] = {"status": "pending", "verdict": None,
                               "uncertainty": None, "context": {},
                               "probes": [], "notes": [],
                               "violations": [], "activities": [],
                               "tickets": {}, "verdict_history": []}
            A["petition"] = {"status": "in_flight", "target": "stage1",
                             "reasons": ev.get("reasons", []),
                             "added": [], "removed": [], "outcome": None}
            why = ", ".join(str(r.get("kind", "?")).replace("_", " ")
                            for r in ev.get("reasons", []))
            A["activities"].append(f"PETITION to Stage 1: re-perceiving — "
                                   f"unresolved: {why}")
            d["activity"] = {"text": f"STAGE 2 petitions STAGE 1: "
                                     f"re-perceiving ({why})",
                             "oid": None, "busy": True}
        elif t == "petition_done":
            P = A.get("petition") or {}
            P.update(status="merged", added=ev.get("added", []),
                     removed=ev.get("removed", []),
                     disputed=ev.get("disputed", []),
                     note=ev.get("note"))
            A["petition"] = P
            A["activities"].append(
                f"petition merged: +{ev.get('added')} "
                f"-{ev.get('removed')} "
                f"({ev.get('n_petitioned', 0)} petitioned entit(ies))")
            if not ev.get("added"):
                # nothing was admitted -> the run stayed on the first record
                _restore_epoch0_perception(d)
        elif t == "petition_failed":
            A["petition"] = {**(A.get("petition") or {}),
                             "status": "failed",
                             "error": ev.get("error", "")}
            A["activities"].append(f"petition FAILED: {ev.get('error', '')[:60]}"
                                   f" — proceeding on the original record")
            _restore_epoch0_perception(d)
        elif t == "petition_outcome":
            P = A.get("petition") or {}
            P["outcome"] = {"resolved": ev.get("resolved"),
                            "before": ev.get("violations_before", []),
                            "after": ev.get("violations_after", [])}
            A["petition"] = P
            how = ("the fresh re-ask" if P.get("target") == "stage2"
                   else "re-perception")
            verdict = (f"pressure RESOLVED by {how}"
                       if ev.get("resolved")
                       else f"pressure remains after {how}")
            A["activities"].append(f"petition outcome: {verdict}")
            d["activity"] = {"text": f"petition: {verdict}",
                             "oid": None, "busy": False}
        elif t == "assess_runoff":
            A["runoff"] = {k: ev.get(k) for k in
                           ("winner", "raw", "top1_votes", "top2_votes",
                            "top1_line", "top2_line")}
            A["activities"].append(
                f"runoff: judge preferred the {ev.get('winner')} reading "
                f"({ev.get('top1_votes')} vs {ev.get('top2_votes')})")
        elif t == "assess_runoff_error":
            A["activities"].append(f"runoff judge unavailable: "
                                   f"{ev.get('error', '')[:60]}")
        elif t == "assess_probe_error":
            A["activities"].append(f"probe {int(ev.get('index', 0)) + 1} "
                                   f"failed: {ev.get('error', '')}")
        elif t == "assess_uncertainty":
            d["activity"] = {"text": f"stability measured: U = "
                                     f"{ev.get('score')}",
                             "oid": None, "busy": False}
            A["uncertainty"] = {k: ev.get(k) for k in
                                ("score", "n_probes", "scenario_agreement",
                                 "type_agreement", "bucket_agreement",
                                 "granular", "drivers", "explanation",
                                 "explainer")}
            A["activities"].append(f"measured U={ev.get('score')} "
                                   f"(drivers: "
                                   f"{', '.join(ev.get('drivers', [])) or 'none'})")
        elif t == "reflect_round_started":
            A.setdefault("reflection", {"rounds": [], "stopped": None,
                                        "u_before": None, "u_after": None})
            trigs = ev.get("triggers", [])

            def _tglabel(tg):
                if tg.get("type") == "violation":
                    return str(tg.get("kind", "?"))
                if tg.get("type") == "membership_split":
                    return f"{tg.get('object_id')} {tg.get('votes')}"
                if tg.get("type") == "field_instability":
                    return str(tg.get("driver", "?"))
                if tg.get("type") == "candidate_runoff":
                    return f"runoff advice ({tg.get('winner')})"
                return str(tg.get("type", "?"))
            summary = ", ".join(_tglabel(tg) for tg in trigs)
            A["reflection"]["rounds"].append(
                {"round": ev.get("round"), "triggers": trigs,
                 "summary": summary, "changed": None,
                 "instruction": ev.get("instruction", "")})
            for tk in A["tickets"].values():
                if tk["status"] == "open":
                    tk["status"] = "fixing"
            A["activities"].append(f"reflection round {ev.get('round')}: "
                                   f"{ev.get('n_triggers')} problem(s) — "
                                   f"{summary}")
            d["activity"] = {"text": f"reflecting: {summary}",
                             "oid": None, "busy": True}
        elif t == "reflect_round_done":
            R = A.get("reflection") or {}
            if R.get("rounds"):
                R["rounds"][-1]["changed"] = ev.get("changed")
                R["rounds"][-1]["violations_after"] = ev.get("violations_after")
            remaining = set(ev.get("violations_after_kinds") or [])
            for tk in A["tickets"].values():
                if tk["status"] in ("open", "fixing"):
                    tk["status"] = "open" if tk["kind"] in remaining else "fixed"
            verdict = ("model revised its answer" if ev.get("changed")
                       else "model stood its ground")
            A["activities"].append(f"reflection round {ev.get('round')}: "
                                   f"{verdict}, "
                                   f"{ev.get('violations_after', 0)} "
                                   f"problem(s) remain")
        elif t == "reflect_error":
            A["activities"].append(f"reflection failed: {ev.get('error')}")
        elif t == "reflect_stopped":
            A.setdefault("reflection", {"rounds": [], "stopped": None,
                                        "u_before": None, "u_after": None})
            A["reflection"].update(stopped=ev.get("reason"),
                                   u_before=ev.get("u_before"),
                                   u_after=ev.get("u_after"))
            for tk in A["tickets"].values():
                if tk["status"] in ("open", "fixing"):
                    tk["status"] = ("fixed" if ev.get("reason") == "clean"
                                    else "stood")
            stamp = {"clean": "all problems resolved",
                     "no_change": "model stood its ground",
                     "cap_reached": "round cap reached",
                     "model_error": "model unavailable"}.get(
                ev.get("reason"), ev.get("reason", ""))
            d["activity"] = {"text": f"reflection: {stamp}", "oid": None,
                             "busy": False}
            if ev.get("rounds"):
                A["activities"].append(f"reflection stopped: {stamp}")
            if ev.get("u_after") is not None:
                A["activities"].append(
                    f"uncertainty U {ev.get('u_before')} -> "
                    f"{ev.get('u_after')} after reflection")
        elif t == "stage_done" and ev.get("stage") == "assess":
            A["status"] = "done"
            v = A.get("verdict") or {}
            if v:
                d["activity"] = {"text": f"assessment done: "
                                         f"{v.get('scenario')} · "
                                         f"{v.get('disaster_type')} · "
                                         f"level {v.get('level')}",
                                 "oid": None, "busy": False}

        # Per-station narration lines.
        def act(stage: str, text: str) -> None:
            d["stage_activities"][stage].append(text)

        if t == "stage_started":
            s = ev.get("stage")
            opener = {"Perceive": "asking the VLM to name every entity...",
                      "Repair": "checking the answer against the rulebook...",
                      "Ground": "detector searching for the named labels...",
                      "Bind": "matching candidates to the model's anchors...",
                      "Mask": "tracing outlines with SAM...",
                      "Assemble": "validating and writing the record..."}.get(s)
            if s in d["stage_activities"] and opener:
                act(s, opener)
        elif t == "stage_done":
            s = ev.get("stage")
            closer = {"Perceive": f"first answer: {ev.get('n_entities', '?')} entities",
                      "Ground": f"{ev.get('n_candidates', '?')} candidate boxes found",
                      "Bind": (f"{ev.get('matched', 0)} matched, "
                               f"{ev.get('fallback', 0)} via SAM fallback"),
                      "Mask": f"{ev.get('masks', '?')} masks captured",
                      "Assemble": "record validated and saved"}.get(s)
            if s in d["stage_activities"] and closer:
                act(s, closer)
        elif t == "violation_found":
            act("Repair", f"found: {str(ev.get('kind', '')).replace('_', ' ')} "
                          f"('{ev.get('raw_label')}')")
        elif t == "repair_round_started":
            d["round_in_progress"] = True
            act("Repair", f"round {ev.get('round')}: asking the model to fix "
                          f"{ev.get('open_violations')} problem(s)...")
        elif t == "repair_round_done":
            d["round_in_progress"] = False
            changed = "model revised its answer" if ev.get("changed") \
                else "model returned the same answer"
            act("Repair", f"round {ev.get('round')} done: {changed}, "
                          f"{ev.get('remaining_violations', 0)} problem(s) remain")
        elif t == "repair_stopped":
            d["round_in_progress"] = False
            act("Repair", {"clean": "all problems resolved — clean",
                           "no_change": "model stood its ground — stopping",
                           "cap_reached": "round cap reached — stopping"}.get(
                ev.get("reason"), ev.get("reason", "")))
        elif t == "masking_entity":
            act("Mask", f"masking {ev.get('object_id')}...")
        elif t == "entity_bound":
            src = ("matched" if ev.get("box_source") == "dino_matched"
                   else "SAM fallback")
            act("Bind", f"{ev.get('object_id')}: {src} "
                        f"(conf {ev.get('confidence', 0):.2f})")

        # Activity ribbon: a one-liner for the scene, updated per event.
        # (assess events set their own ribbon text in the block above; the
        # generic handler must not blank it.)
        if t == "stage_started" and ev.get("stage") != "assess":
            text = {"Perceive": "model reading the scene...",
                    "Repair": "rulebook checking the answer...",
                    "Ground": "detector searching for the named entities...",
                    "Bind": "binding boxes to the model's anchors...",
                    "Mask": "SAM tracing entity outlines...",
                    "Assemble": "assembling the record..."}.get(ev.get("stage"), "")
            d["activity"] = {"text": text, "oid": None,
                             "busy": ev.get("stage") in ("Perceive", "Repair")}
        elif t == "violation_found":
            d["activity"] = {"text": f"violation: {ev.get('kind', '').replace('_', ' ')} "
                                     f"('{ev.get('raw_label')}')", "oid": None, "busy": True}
        elif t == "repair_round_started":
            d["activity"] = {"text": f"repair round {ev.get('round')}: asking the model "
                                     f"to fix {ev.get('open_violations')} problem(s)...",
                             "oid": None, "busy": True}
        elif t == "repair_stopped":
            verdict = {"clean": "repair clean", "no_change": "model stood its ground",
                       "cap_reached": "repair cap reached"}.get(ev.get("reason"), "")
            d["activity"] = {"text": verdict, "oid": None, "busy": False}
        elif t == "entity_bound":
            src = "matched" if ev.get("box_source") == "dino_matched" else "SAM fallback"
            d["activity"] = {"text": f"{ev.get('object_id')} bound ({src}, "
                                     f"conf {ev.get('confidence', 0):.2f})",
                             "oid": ev.get("object_id"), "busy": False}
        elif t == "masking_entity":
            d["activity"] = {"text": f"masking {ev.get('object_id')}...",
                             "oid": ev.get("object_id"), "busy": True}
        elif t == "assembled":
            d["activity"] = {"text": "done", "oid": None, "busy": False}
    # Judge wall-clock for the bench header: everything between the last
    # deterministic eval (alignment) and trust is judge time. Live events
    # carry wall stamps; replayed ones do not, so this is best-effort.
    _ts = {e.get("type"): e.get("t") for e in events
           if isinstance(e, dict) and e.get("t")}
    if _ts.get("alignment_ready") and _ts.get("trust_ready"):
        d["s4_judge_seconds"] = round(_ts["trust_ready"]
                                      - _ts["alignment_ready"], 1)
    return d


# ── Replay: saved record -> synthetic event stream ──────────────────────


def build_replay_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct a plausible event stream from a frozen PerceptionResult.

    Timings are unknown for saved records; the sequence is faithful, the
    clock is not. Repair rounds come from the stored repair_trace; when a
    loop stopped unclean, the remaining set is approximated by the last
    round's violations (the trace stores what each round SAW)."""
    ev: list[dict[str, Any]] = []
    ev.append({"type": "run_started",
               "image_size": record.get("image_size"),
               "caption": record.get("caption", ""),
               "image_name": Path(record.get("image_path", "scene")).name})
    objs = record.get("detected_objects", [])
    ev.append({"type": "stage_started", "stage": "Perceive"})
    # Saved records hold the post-repair list; the true first answer is only
    # known live. Close enough for replay's Perceive detail line.
    ev.append({"type": "stage_done", "stage": "Perceive", "seconds": None,
               "n_entities": len(objs),
               "entities": [{"label": o.get("label"), "state": o.get("state")}
                            for o in objs]})
    ev.append({"type": "stage_started", "stage": "Repair"})
    trace = record.get("repair_trace") or {}
    rounds = trace.get("rounds", [])
    for r in rounds:
        for v in r.get("violations", []):
            ev.append({"type": "violation_found", "round": r.get("round_number", 1), **v})
        ev.append({"type": "repair_round_done", "round": r.get("round_number", 1),
                   "changed": r.get("changed", True),
                   "entities": r.get("entities_after", [])})
    reason = trace.get("stopped_reason") or "clean"
    remaining = (rounds[-1].get("violations", [])
                 if rounds and reason != "clean" else [])
    ev.append({"type": "repair_stopped", "reason": reason,
               "rounds": len(rounds), "remaining": remaining})
    ev.append({"type": "stage_done", "stage": "Repair", "seconds": None,
               "stopped": reason})
    ev.append({"type": "anchors_ready", "entities": [
        {k: o.get(k) for k in ("object_id", "label", "state", "description", "anchor_bbox")}
        for o in objs
    ]})
    for st, extra in (("Ground", {}),
                      ("Bind", {"matched": sum(1 for o in objs if o.get("box_source") == "dino_matched"),
                                "fallback": sum(1 for o in objs if o.get("box_source") == "vlm_sam_fallback")})):
        ev.append({"type": "stage_started", "stage": st})
        ev.append({"type": "stage_done", "stage": st, "seconds": None, **extra})
    for o in objs:
        ev.append({"type": "entity_bound", "object_id": o.get("object_id"),
                   "box_source": o.get("box_source"), "bbox": o.get("bbox"),
                   "confidence": o.get("box_confidence", 0.0)})
    ev.append({"type": "stage_started", "stage": "Mask"})
    ev.append({"type": "stage_done", "stage": "Mask", "seconds": None,
               "masks": sum(1 for o in objs if o.get("mask_path"))})
    ev.append({"type": "stage_started", "stage": "Assemble"})
    ev.append({"type": "stage_done", "stage": "Assemble", "seconds": None,
               "n_entities": len(objs)})
    ev.append({"type": "assembled", "result": record})
    return ev


def saved_records() -> list[dict[str, str]]:
    """Replayables: frozen worked-example records, plus every UI run's
    events.jsonl flight recorder (true replay, exact event stream)."""
    opts: list[dict[str, str]] = []
    if PERCEPTION_DIR.exists():
        opts += [{"label": p.name.replace("__perception.json", ""), "value": str(p)}
                 for p in sorted(PERCEPTION_DIR.glob("*__perception.json"))]
    if UI_RUNS_DIR.exists():
        opts += [{"label": f"ui run · {p.parent.name}", "value": str(p)}
                 for p in sorted(UI_RUNS_DIR.glob("*/events.jsonl"))]
    return opts


def image_src_for_record(json_path: Path) -> str | None:
    stem = json_path.name.replace("__perception.json", "")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = SCENES_DIR / f"{stem}{ext}"
        if p.exists():
            mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


# ── Live run (background thread) ────────────────────────────────────────


UI_RUNS_DIR = REPO_ROOT / "exports" / "agentic_runs"


def make_event_sink(run_id: str, run_dir: Path):
    """Event sink: append to the in-memory stream AND to events.jsonl.

    events.jsonl is the run's flight recorder: one JSON object per line,
    timestamped, written as it happens. It is the durable record the
    scrubber, future replays, and the dialogue agent read; the in-memory
    list only serves the live screen."""
    import time as _time

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    def sink(event: dict[str, Any]) -> None:
        stamped = {"t": round(_time.time(), 3), **event}
        RUNS[run_id]["events"].append(stamped)
        with log_path.open("a") as f:
            f.write(json.dumps(stamped) + "\n")

    return sink


def start_live_run(image_bytes: bytes, filename: str, caption: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    run_dir = UI_RUNS_DIR / f"ui_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_dir / (filename or "scene.jpg")
    image_path.write_bytes(image_bytes)
    (run_dir / "caption.txt").write_text(caption or "")
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    RUNS[run_id] = {
        "events": [],
        "image_src": f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}",
        "done": False, "error": None,
        "record_name": f"ui_{run_id}",     # how agent_tools will know this run
    }
    sink = make_event_sink(run_id, run_dir)

    def worker() -> None:
        try:
            from agentic.assessment import (DEFAULT_N_PROBES, run_assessment)
            from agentic.perception import run_perception
            from agentic.retrieval import drain_shadow_log
            from agentic.uncertainty import _ollama_explain
            drain_shadow_log()                    # clear any prior run's rows
            record = run_perception(image_path, caption=caption,
                                    out_dir=run_dir, on_event=sink)
            # Stage 1 (repair P1-P6) quotes rules too, through the same
            # retrieval switch — so in rag/both mode it fills the shadow.
            # Drain it HERE, before Stage 2 runs, or Stage 2's drain would
            # swallow these rows and Stage 1 would show nothing.
            perceive_shadow = drain_shadow_log()
            if perceive_shadow:
                from agentic.rulebook_rag import last_backend_reason
                sink({"type": "retrieval_shadow", "stage": "perceive",
                      "rows": perceive_shadow,
                      "backend_reason": last_backend_reason()})
            # keep the assessed record so the on-screen RAG-diff button can
            # re-run Stage 2 both ways on it later.
            RUNS[run_id]["record"] = record.model_copy(deep=True)
            # Stage 2, full story: assess -> reflection -> (petition ->
            # cascade re-assessment) — the one ledgered image look-back.
            try:
                from agentic.evals import _ollama_judge as _judge
            except Exception:
                _judge = None
            from agentic.graph_live import assess_with_control
            record, result, petitioned = assess_with_control(
                str(image_path), record, on_event=sink,
                n_probes=DEFAULT_N_PROBES,
                explain_fn=_ollama_explain,
                runoff_judge_fn=_judge)
            # Surface the RAG shadow (only populated in rag/both mode): the
            # exact-key vs RAG top-1 comparison for the rules this run used.
            shadow = drain_shadow_log()
            if shadow:
                from agentic.rulebook_rag import last_backend_reason
                sink({"type": "retrieval_shadow", "stage": "assess",
                      "rows": shadow,
                      "backend_reason": last_backend_reason()})
                # AUTO result diff (Both/RAG mode): only when a fired rule
                # actually differs does the answer have any chance to move —
                # if every fired rule agrees, the two runs are identical by
                # construction, so we skip the extra model passes. When they
                # DO differ, re-run Stage 2 both ways (probes off) and show
                # whether RAG's rule choice moved the final answer. No button.
                from agentic.retrieval import retrieval_mode
                if (retrieval_mode() in ("both", "rag")
                        and any(not r.get("agree") for r in shadow)):
                    try:
                        from agentic.retrieval import (compare_retrieval_modes,
                                                       drain_shadow_log as _dr)
                        _dr()                     # keep compare out of shadow
                        diff = compare_retrieval_modes(
                            RUNS[run_id]["record"], n_probes=0)
                        _dr()
                        sink({"type": "retrieval_result_diff",
                              "changed": diff["changed"],
                              "rulebook_line": diff["rulebook_line"],
                              "rag_line": diff["rag_line"]})
                    except Exception as exc:
                        sink({"type": "retrieval_result_diff",
                              "error": str(exc)[:200]})
            if petitioned:
                (run_dir / "perception_petitioned.json").write_text(
                    record.model_dump_json(indent=2))
            (run_dir / "assessment.json").write_text(result.model_dump_json(indent=2))
            # PAIRWISE JUDGE policy: auto-run exactly when reflection
            # CHANGED the answer — that is the only moment a pre-vs-post
            # comparison exists. Unchanged run -> nothing to judge; high
            # U alone is the probe meter's business, not the pairwise
            # judge's. Best-effort: a missing judge model never breaks
            # the run (the card just says "not yet run").
            tr = result.reflection_trace or {}
            if any(r.get("changed") for r in tr.get("rounds", [])):
                RUNS[run_id]["judge"] = {"status": "running"}
                try:
                    from agentic.assessment import parse_assessment
                    from agentic.evals import judge_pairwise, substantive_key
                    pre, _n = parse_assessment(result.raw_answer)

                    def _vline(v):
                        th = ",".join(t.object_id for t in v.threats) or "-"
                        return (f"{v.disaster_scenario}·"
                                f"L{v.disaster_level} threats[{th}]")
                    pre_d = pre.model_dump()
                    post_d = result.assessment.model_dump()
                    # SUBSTANTIVE-CHANGE GATE (C_tanker ui_529ce417):
                    # prose-only reflection changes give the judge an
                    # undefined question — a forced choice between
                    # identical decisions is noise wearing an F4
                    # costume. Skip, and say so on the card.
                    if substantive_key(pre_d) == substantive_key(post_d):
                        sink({"type": "pairwise_skipped",
                              "reason": "no substantive change "
                                        "(narration only)"})
                        RUNS[run_id]["judge"] = {
                            "winner": "skipped",
                            "raw": "reflection changed narration only — "
                                   "decision layer identical, nothing "
                                   "to adjudicate",
                            "pre_line": _vline(pre),
                            "post_line": _vline(result.assessment)}
                    else:
                        j = judge_pairwise(
                            pre_d, post_d,
                            record.model_dump(),
                            RUNS[run_id]["record_name"])
                        j["pre_line"] = _vline(pre)
                        j["post_line"] = _vline(result.assessment)
                        RUNS[run_id]["judge"] = j
                except Exception as exc:
                    RUNS[run_id]["judge"] = {"winner": "error",
                                             "raw": str(exc)[:120]}
            # ── Stage 4 · recommendations (Phase 1a spine) ──────────────
            # Runs after Stage 2 is final: recommend -> Graph A -> Graph B ->
            # pick targets. Image is bound into the query_fn as context (ids
            # stay frozen — no re-perception). Best-effort: a Stage-4 failure
            # surfaces as an event and never kills the Stage 1-2 screen.
            try:
                import base64
                import mimetypes as _mt

                from agentic.graph_s4 import stage4_with_control
                from agentic.recommend import (DEFAULT_REC_N_PROBES,
                                               REC_PROBE_TEMPERATURE, _query_vlm)
                _mime = _mt.guess_type(str(image_path))[0] or "image/jpeg"
                _data_url = (f"data:{_mime};base64,"
                             f"{base64.b64encode(image_path.read_bytes()).decode()}")

                def _s4_query(prompt, _u=_data_url):
                    return _query_vlm(prompt, image_contents=_u, temperature=0.0)

                # probe re-asks: same image, raised temperature (channel-2 U)
                def _s4_probe(prompt, _u=_data_url):
                    return _query_vlm(prompt, image_contents=_u,
                                      temperature=REC_PROBE_TEMPERATURE)
                _s4_n_probes = int(os.getenv("REC_N_PROBES",
                                             str(DEFAULT_REC_N_PROBES)))
                # F24 — the card judge. ADVISORY and display-only: it never
                # enters a score, which is exactly why it is safe to leave on
                # during calibration. A judge that cannot reach its model
                # degrades to 'unclear' per card, never a failed run.
                try:
                    from agentic.judge_card import (JUDGE_PROBE_TEMPERATURE,
                                                    _ollama_judge)

                    def _card_judge(p):
                        return _ollama_judge(
                            p, temperature=JUDGE_PROBE_TEMPERATURE)
                except Exception:
                    _card_judge = None
                if os.getenv("S4_CARD_JUDGE", "1") != "1":
                    _card_judge = None
                s4 = stage4_with_control(record, result.assessment,
                                         str(image_path), query_fn=_s4_query,
                                         probe_fn=_s4_probe,
                                         judge_fn=_card_judge,
                                         n_probes=_s4_n_probes,
                                         on_event=sink)
                (run_dir / "stage4.json").write_text(s4.model_dump_json(indent=2))
                RUNS[run_id]["stage4"] = s4.model_dump()
                sink({"type": "stage4_result", "result": s4.model_dump()})
            except Exception as exc:
                sink({"type": "stage4_error", "message": str(exc)[:200]})
        except Exception as exc:  # surfaced as a card, never a dead screen
            sink({"type": "run_error", "message": str(exc)})
            RUNS[run_id]["error"] = str(exc)
        finally:
            RUNS[run_id]["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return run_id


def start_replay(json_path: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    path = Path(json_path)
    if path.name == "events.jsonl":
        # True replay: the exact recorded event stream of a past UI run.
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        image_src = None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            for img in sorted(path.parent.glob(f"*{ext}")):
                if "__" not in img.name:      # skip overlays/masks
                    mime = mimetypes.guess_type(str(img))[0] or "image/jpeg"
                    image_src = (f"data:{mime};base64,"
                                 f"{base64.b64encode(img.read_bytes()).decode()}")
                    break
            if image_src:
                break
        RUNS[run_id] = {"events": events, "image_src": image_src,
                        "done": True, "error": None,
                        "record_name": path.parent.name}
        return run_id
    record = json.loads(path.read_text())
    name = path.name.replace("__perception.json", "")
    events = build_replay_events(record)
    events += build_assess_replay_events(name, record)
    RUNS[run_id] = {
        "events": events,
        "image_src": image_src_for_record(path),
        "done": True, "error": None,
        "record_name": name,
    }
    return run_id


def build_assess_replay_events(name: str,
                               record: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize Stage 2 events from a frozen assessment record (written
    by `python -m agentic.assessment`), so replaying a frozen scene shows
    PHASE 1 exactly as a live run would. Probe-by-probe order is not
    stored; the verdict, violations, and measured uncertainty are."""
    path = ASSESSMENT_DIR / f"{name}__assessment.json"
    if not path.exists():
        return []
    try:
        rec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    a = rec.get("assessment", {})
    mu = rec.get("measured_uncertainty")
    objs = record.get("detected_objects", [])
    from agentic.geometry import spatial_hints as _sh2
    ev: list[dict[str, Any]] = [
        {"type": "stage_started", "stage": "assess"},
        {"type": "assess_context", "n_entities": len(objs),
         "hazard_ids": [o.get("object_id") for o in objs
                        if o.get("state_kind") == "hazard_bearing"],
         "at_risk_ids": [o.get("object_id") for o in objs
                         if o.get("state_kind") == "at_risk"],
         "n_spatial_hints": len(_sh2(objs, record.get("image_size"))),
         "spatial_pairs": [f"{h['other']} ↔ {h['hazard']} "
                           f"({h['relation']}, {h['gap_px']:.0f}px)"
                           for h in _sh2(objs, record.get("image_size"))]},
    ]
    for note in rec.get("parse_notes", []):
        ev.append({"type": "assess_parse_note", "note": note})
    for v in rec.get("violations", []):
        ev.append({"type": "assess_violation", **v})
    ev.append({"type": "assess_verdict",
               "scenario": a.get("disaster_scenario"),
               "disaster_type": a.get("disaster_type"),
               "level": a.get("disaster_level"),
               "bucket": a.get("severity_bucket"),
               "self_confidence": a.get("self_confidence"),
               "threats": a.get("threats", []),
               "at_risk": a.get("at_risk", []),
               "n_violations": len(rec.get("violations", []))})
    if mu:
        ev.append({"type": "assess_uncertainty",
                   "score": mu.get("score"), "n_probes": mu.get("n_probes"),
                   "granular": mu.get("granular") or {},
                   "scenario_agreement": mu.get("scenario_agreement"),
                   "type_agreement": mu.get("type_agreement"),
                   "bucket_agreement": mu.get("bucket_agreement"),
                   "drivers": [d.get("kind") for d in mu.get("drivers", [])],
                   "explanation": mu.get("explanation", ""),
                   "explainer": mu.get("explainer", "")})
    trace = rec.get("reflection_trace") or {}
    for r in trace.get("rounds", []):
        trigs = r.get("triggers", [])
        ev.append({"type": "reflect_round_started",
                   "round": r.get("round_number"),
                   "triggers": trigs,
                   "instruction": r.get("instruction", ""),
                   "n_triggers": len(trigs)})
        ev.append({"type": "reflect_round_done",
                   "round": r.get("round_number"),
                   "changed": r.get("changed"),
                   "violations_after": len(r.get("violations_after", [])),
                   "violations_after_kinds": [v.get("kind") for v in
                                              r.get("violations_after", [])]})
    if trace:
        ev.append({"type": "reflect_stopped",
                   "reason": trace.get("stopped_reason"),
                   "rounds": len(trace.get("rounds", [])),
                   "u_before": trace.get("u_before"),
                   "u_after": trace.get("u_after")})
    ev.append({"type": "stage_done", "stage": "assess"})
    return ev


# ── Components (each a pure function of the derived state) ──────────────


def rail_component(d: dict[str, Any]) -> html.Div:
    """The stage rail. Perceive and Repair carry rich detail lines:
    Perceive lists what the model's FIRST answer named (label·state, so a
    later repair is visible as a difference); Repair summarizes the loop's
    ledger (violations by kind, fixed vs stood, rounds, verdict)."""
    items = []
    for s in STAGES:
        st = d["stages"][s]
        cls = {"pending": "station", "active": "station active",
               "done": "station done"}[st["status"]]
        cls += f" st-{s.lower()}"          # per-stage tint (shaded cards)
        secs = f"{st['seconds']:.1f}s" if st.get("seconds") else ""
        # Compact status shown in the collapsed header.
        if s == "Repair" and d["violations"]:
            compact = f"{len(d['violations'])} violation(s)"
        elif st.get("info"):
            compact = st["info"]
        else:
            compact = ""
        head = html.Summary([html.Span(s, className="station-name"),
                             html.Span(secs, className="station-secs"),
                             html.Span(compact, className="station-compact"),
                             html.Span("▾", className="station-chev")],
                            className="station-head")
        body: list[Any] = [head]

        if s == "Perceive" and d["perceive_entities"]:
            named = []
            for e in d["perceive_entities"][:8]:
                lbl = str(e.get("label", "?"))
                stt = str(e.get("state", ""))
                named.append(f"{lbl}·{stt}" if stt else lbl)
            more = len(d["perceive_entities"]) - 8
            line = ", ".join(named) + (f" +{more} more" if more > 0 else "")
            body.append(html.Div(f"first answer ({len(d['perceive_entities'])}): {line}",
                                 className="station-detail"))
        elif s == "Perceive":
            body.append(html.Div(st.get("info", ""), className="station-info"))

        if s == "Repair":
            n_v = len(d["violations"])
            fixed = sum(1 for v in d["violations"].values() if v["status"] == "fixed")
            stood = sum(1 for v in d["violations"].values() if v["status"] == "stood")
            open_ = n_v - fixed - stood
            pips = "".join("●" if i < d["rounds_used"] else "○" for i in range(2))
            stamp = {"clean": "CLEAN", "no_change": "STOOD GROUND",
                     "cap_reached": "CAP REACHED", "skipped": "SKIPPED", None: ""}.get(
                d["stop_reason"], "")
            if n_v:
                kinds: dict[str, int] = {}
                for v in d["violations"].values():
                    kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
                kind_line = " · ".join(f"{k.replace('_', ' ')} ×{n}"
                                       for k, n in kinds.items())
                counts = f"{n_v} violation(s): {fixed} fixed"
                if stood:
                    counts += f", {stood} stood"
                if open_:
                    counts += f", {open_} open"
                body.append(html.Div(counts, className="station-detail"))
                body.append(html.Div(kind_line, className="station-detail dim"))
            elif st["status"] != "pending":
                body.append(html.Div("no violations in the first answer",
                                     className="station-detail"))
            body.append(html.Div(f"loop {pips}  {stamp}", className="station-loop"))

        if s not in ("Perceive", "Repair"):
            body.append(html.Div(st.get("info", ""), className="station-info"))

        # Activity timeline: a mini vertical timeline inside the card
        # (nodes on a connector line, one per activity; the newest node of
        # an active stage pulses as "happening now"). A striking variation
        # of the Cowork activity feed, per Sunny, not a copy: nodes carry
        # outcome glyphs and the stage's color, the connector is drawn in
        # the stage tint, and the live node radiates.
        acts = d["stage_activities"].get(s, [])
        if acts:
            color = STAGE_COLORS.get(s, "#64748b")
            shown = acts[-5:]
            rows: list[Any] = []
            if len(acts) > 5:
                rows.append(html.Div([
                    html.Div("⋯", className="tl-node dim",
                             style={"borderColor": "#cbd5e1"}),
                    html.Div(f"{len(acts) - 5} earlier steps",
                             className="tl-text dim"),
                ], className="tl-row"))
            for i, a in enumerate(shown):
                now = (st["status"] == "active" and i == len(shown) - 1)
                glyph, mod = _timeline_glyph(a)
                node_style = {"borderColor": color}
                if mod == "warn":
                    node_style = {"borderColor": "#f59e0b", "color": "#b45309"}
                if now:
                    glyph, mod = "▸", "now"
                    node_style = {"borderColor": color, "color": color,
                                  "boxShadow": f"0 0 0 4px {color}22"}
                rows.append(html.Div([
                    html.Div(glyph, className=f"tl-node {mod}", style=node_style),
                    html.Div(a, className=f"tl-text {mod}"),
                ], className="tl-row"))
            body.append(html.Div(rows, className="tl",
                                 style={"--tl-line": f"{color}55"}))

        # Collapsible station: active stages arrive open; finished ones can
        # be folded to their header line. (The render cache below keeps the
        # user's toggles from being reset by the refresh interval.)
        items.append(html.Details(body, className=cls,
                                  open=(st["status"] != "pending")))

    # FINAL PERCEPTION list (Sunny 2026-07-22: "the right side panel
    # needs to show what entities and their states after the final
    # perception step — currently we only show on image"). Every entity
    # the record ends with, plain: id · state, colored by kind.
    res = d.get("result")
    if res and res.get("detected_objects"):
        rows: list[Any] = [html.Div(
            "FINAL PERCEPTION — what the scene contains",
            className="unc-tag", style={"marginTop": "8px"})]
        for o in res["detected_objects"]:
            color = STATE_KIND_CSS.get(o.get("state_kind"), "#94a3b8")
            extra = ""
            if o.get("state_note"):
                extra = " (state derived in code)"
            elif o.get("provenance") == "petition":
                extra = " (added by petition)"
            rows.append(html.Div([
                html.Span("●", style={"color": color,
                                      "marginRight": "6px"}),
                html.Span(o.get("object_id", "?"),
                          style={"fontWeight": "700"}),
                html.Span(f" · {o.get('state')}{extra}",
                          style={"color": "#475569",
                                 "fontSize": "12px"}),
            ], style={"padding": "1px 0"}))
        for n in res.get("notes") or []:
            if n.startswith("duplicate merged"):
                rows.append(html.Div(f"⚭ {n}",
                                     style={"fontSize": "11px",
                                            "color": "#b45309"}))
        items.append(html.Div(rows, className="unc-panel",
                              style={"borderColor": "#16a34a"}))
    return html.Div(items, className="rail")


def _verdict_to_assessment(v: dict[str, Any]) -> dict[str, Any]:
    """Map an assess_verdict event payload to eval_stage2's input shape."""
    return {"disaster_scenario": v.get("scenario"),
            "disaster_type": v.get("disaster_type"),
            "severity_bucket": v.get("bucket"),
            "threats": v.get("threats") or [],
            "at_risk": v.get("at_risk") or []}


def gt_eval_overlay(d: dict[str, Any],
                    record_name: str | None) -> dict[str, Any] | None:
    """DEV-ONLY overlay: GT-based scoring for calibration scenes. Returns
    None when no GT exists for this scene — which is every production
    scene; the overlay is a lab instrument, never a shipped feature."""
    try:
        from agentic.evals import eval_stage1, eval_stage2, load_gt, quadrant
        gt = load_gt()
    except Exception:
        return None
    matched_by = None
    if record_name in gt:
        matched_by = "scene"
    else:
        stem = Path(str(d.get("image_name") or "")).stem
        if stem in gt:
            record_name, matched_by = stem, "image_name"
    if matched_by is None:
        return None
    hist = d["assess"].get("verdict_history") or []
    if not hist:
        return None
    pre = eval_stage2(_verdict_to_assessment(hist[0]), gt[record_name]["stage2"])
    post = eval_stage2(_verdict_to_assessment(hist[-1]), gt[record_name]["stage2"])
    s1 = (eval_stage1(d["result"], gt[record_name]["stage1"])
          if d.get("result") else None)
    R = d["assess"].get("reflection") or {}
    return {"pre": pre, "post": post, "stage1": s1,
            "matched_by": matched_by,
            "quadrant": quadrant(pre["error_score"], post["error_score"],
                                 R.get("u_before"), R.get("u_after"))}


def rag_shadow_panel(shadow: list[dict[str, Any]] | None, stage_label: str,
                     *, result_diff: dict[str, Any] | None = None,
                     backend_reason: str | None = None):
    """One RAG-shadow panel: exact-key vs RAG top-1 for the rules a stage
    quoted, deduped with a ×N count. Used by BOTH Stage 1 (repair) and
    Stage 2 (assess). result_diff is Stage-2-only — the auto answer diff
    shown when a fired rule differs. backend_reason, when the engine is the
    word-count fallback, explains WHY the real embedding didn't run.
    Returns an html.Div, or None if the stage quoted no rules."""
    if not shadow:
        return None
    # dedupe: one row per (kind, exact, rag), with a ×N count — a rule
    # quoted for several entities shouldn't repeat.
    uniq: dict[tuple, dict] = {}
    for r in shadow:
        key = (r.get("kind"), r.get("exact_rule_id"), r.get("rag_kind"))
        if key in uniq:
            uniq[key]["_n"] += 1
        else:
            uniq[key] = {**r, "_n": 1}
    urows = list(uniq.values())
    n = len(urows)
    agree = sum(1 for r in urows if r.get("agree"))
    eng = urows[0].get("backend") or "?"
    # name the engine plainly: real embedding vs the word-count fallback
    eng_plain = ("real embedding (semantic)" if eng == "llamaindex_chroma"
                 else "word-count fallback" if eng == "keyword_fallback"
                 else eng)
    header = [html.Div([
        html.Span(f"RAG SHADOW · {stage_label}", className="unc-tag"),
        html.Span(f"  exact-key vs RAG top-1 · {agree}/{n} rules agree · "
                  f"engine: {eng_plain}",
                  style={"fontSize": "11px", "color": "#64748b"}),
    ], style={"marginBottom": "4px"})]
    # When we're on the fallback, say WHY the real embedding didn't run —
    # so a keyword result is never a silent mystery.
    if eng == "keyword_fallback" and backend_reason:
        header.append(html.Div(
            f"⚠ real embedding OFF — {backend_reason}",
            style={"fontSize": "11px", "color": "#b45309",
                   "marginBottom": "4px", "fontStyle": "italic"}))
    rows = header
    for r in urows:
        ok = r.get("agree")
        cnt = (f"  ×{r['_n']}" if r.get("_n", 1) > 1 else "")
        rows.append(html.Div([
            html.Span(f"{r.get('exact_rule_id') or '?'} "
                      f"{r.get('kind')}",
                      style={"fontWeight": "700", "fontSize": "12px"}),
            html.Span("  →  RAG: ", style={"color": "#94a3b8",
                                           "fontSize": "12px"}),
            html.Span(f"{r.get('rag_rule_id') or '-'} "
                      f"{r.get('rag_kind') or '-'}",
                      style={"fontSize": "12px",
                             "color": "#16a34a" if ok else "#b45309"}),
            html.Span((" ✓ match" if ok else " ✗ differs") + cnt,
                      style={"marginLeft": "auto", "fontSize": "11px",
                             "fontWeight": "700",
                             "color": "#16a34a" if ok else "#b45309"}),
        ], style={"display": "flex", "alignItems": "baseline",
                  "padding": "2px 0"}))
    # When fired rules differ, the result diff was computed automatically
    # (both/rag mode). Show it right here — no button, no command line.
    # Stage 1 passes result_diff=None, so it only shows the "differ" count.
    if agree < n:
        rdiff = result_diff
        if rdiff and rdiff.get("error"):
            rows.append(html.Div(
                f"→ {n - agree} rule(s) differ. Result diff could not "
                f"run: {rdiff['error']}",
                style={"fontSize": "11px", "color": "#b45309",
                       "marginTop": "5px", "fontStyle": "italic"}))
        elif rdiff:
            changed = rdiff.get("changed")
            dc = "#dc2626" if changed else "#16a34a"
            head = ("↳ RAG's rule choice MOVED the answer:" if changed
                    else "↳ but the final answer is identical — RAG's "
                         "rule choice did NOT change it:")
            rows.append(html.Div([
                html.Div(f"{n - agree} rule(s) differ. " + head,
                         style={"fontSize": "11px", "color": dc,
                                "fontWeight": "700", "marginBottom": "3px"}),
                html.Div([html.Span("exact-key: ",
                                    style={"color": "#64748b"}),
                          html.Span(rdiff.get("rulebook_line") or "-")],
                         style={"fontSize": "12px"}),
                html.Div([html.Span("RAG      : ",
                                    style={"color": "#64748b"}),
                          html.Span(rdiff.get("rag_line") or "-",
                                    style={"color": dc})],
                         style={"fontSize": "12px"}),
            ], style={"marginTop": "6px", "padding": "6px 8px",
                      "borderTop": "1px dashed #c4b5fd"}))
        elif result_diff is None and stage_label.startswith("Stage 1"):
            # Stage 1 has no verdict to move, so no result diff — just note it.
            rows.append(html.Div(
                f"→ {n - agree} rule(s) differ. (Repair has no verdict to "
                f"move — the answer diff is a Stage 2 check.)",
                style={"fontSize": "11px", "color": "#7c3aed",
                       "marginTop": "5px", "fontStyle": "italic"}))
        else:
            rows.append(html.Div(
                f"→ {n - agree} rule(s) differ. Comparing answers…",
                style={"fontSize": "11px", "color": "#7c3aed",
                       "marginTop": "5px", "fontStyle": "italic"}))
    return html.Div(rows, className="unc-panel",
                    style={"borderColor": "#a78bfa", "background": "#faf5ff"})


def stage4_status_span(d: dict[str, Any]) -> html.Span:
    """The running/done badge on the Stage 4 card header — same look as
    Stages 1-2. Steps: recommend → Graph A → Graph B → pick."""
    # Kept in step with the LIVE list in stage4_component: the judges own
    # their time in BOTH places, or the header chip says "trust · step 6/6"
    # for 20 minutes while the body correctly shows the runoff voting.
    STEPS = [("recommend", "recommend"), ("uncertainty", "uncertainty"),
             ("Graph A", "graph_a"), ("Graph B", "graph_b"),
             ("pick", "picks"), ("card judge", "card_judge"),
             ("graph judge", "graph_judge"), ("runoff twins", "runoff"),
             ("trust", "trust")]
    if d.get("stage4") is not None or d.get("stage4_error"):
        return phase_status_span([(name, "done") for name, _ in STEPS])
    marks = d.get("stage4_marks") or set()
    if not (d.get("stage4_started") or marks):
        return html.Span("○ waiting", className="phase-status pending")
    steps: list[tuple[str, str]] = []
    active_used = False
    for name, key in STEPS:
        if key in marks:
            steps.append((name, "done"))
        elif not active_used:
            steps.append((name, "active"))
            active_used = True
        else:
            steps.append((name, "pending"))
    return phase_status_span(steps)


def _node_chip(nid: str, node: dict[str, Any],
               at_risk: bool = False) -> html.Span:
    """One entity as a colored chip: red = hazard (source of harm),
    amber = at-risk (target), gray = neutral. State shown when present.
    `at_risk` is passed in because Graph B's nodes carry no at_risk flag —
    the caller derives it from the graph structure so both graphs match."""
    haz = bool(node.get("hazardous"))
    atr = bool(node.get("at_risk")) or at_risk
    state = node.get("state", "")
    if haz:
        bg, fg, bd = "#fee2e2", "#b91c1c", "#fca5a5"
    elif atr:
        bg, fg, bd = "#fef3c7", "#b45309", "#fde68a"
    else:
        bg, fg, bd = "#f1f5f9", "#334155", "#e2e8f0"
    return html.Span(f"{nid}{(' · ' + state) if state else ''}",
                     style={"fontSize": "11.5px", "fontWeight": "600",
                            "padding": "1px 6px", "borderRadius": "6px",
                            "background": bg, "color": fg,
                            "border": f"1px solid {bd}",
                            "display": "inline-block", "margin": "2px 3px 0 0"})


def _alignment_edge_row(edge: Any, chipper=None,
                        color: str = "#b45309") -> html.Div:
    """Render one disagreeing causal link as 'source —effect→ target' with the
    real entity ids as chips. `edge` is now an id-level dict {source, effect,
    target}; a legacy tuple/list is still tolerated."""
    if isinstance(edge, dict):
        src, eff, tgt = (str(edge.get("source", "")), str(edge.get("effect", "")),
                         str(edge.get("target", "")))
    else:
        parts = list(edge) if isinstance(edge, (list, tuple)) else [str(edge)]
        src, eff, tgt = (parts + ["", "", ""])[:3]
        src, eff, tgt = str(src), str(eff), str(tgt)

    def _e(x):
        return chipper(x) if chipper else html.Span(x, style={"fontWeight": "600"})
    body = [_e(src), html.Span(f"  —{eff}→  ", style={"color": "#7c3aed"}), _e(tgt)]
    return html.Div(body, style={"fontSize": "11.5px", "color": color,
                                 "paddingLeft": "10px", "lineHeight": "1.9"})


def _resolved_id_rows(aliases: dict, notes: dict | None = None) -> list:
    """F40 — say ONCE, at the top, that the model named an entity the scene
    does not have.

    D_aerial: Graph B called the spill `chemical_spill_1` — the state word with
    `_1` on the end — while the scene entity is `spill_1`. Both graphs said the
    spill endangers the two workers, and because the ids differed the same
    claim counted as a fabrication AND an omission at once. It silently
    corrupted four separate readings on the panel.

    The comparison below now runs on resolved ids. This line exists so a reader
    knows the model's own graph used a name that was not on offer.

    F45: `notes` says WHICH rung of the matcher fired — verbatim, synonym or
    head noun. `chemical_worker_2 → hazmat_worker_2 (head noun, by number)`
    is a looser merge than `chemical_spill_1 → spill_1 (synonym)`, and a
    reader who cannot tell them apart cannot audit either.
    """
    if not aliases:
        return []
    notes = notes or {}

    def _one(k: str, v: str) -> str:
        how = notes.get(k)
        return f"{k} → {v} ({how})" if how else f"{k} → {v}"
    pairs = ", ".join(_one(k, v) for k, v in sorted(aliases.items()))
    return [html.Div(
        f"⚠ the model's graph named entities the scene does not have: {pairs}. "
        f"Compared as the same entity.",
        style={"fontSize": "11px", "color": "#b45309", "margin": "2px 0 4px"})]


def _disagreement_rows(edges: list, title: str, s4: dict,
                       chipper) -> list:
    """F40 — one line per ENTITY PAIR, not per edge, ordered by how dangerous
    the hazard is.

    D_aerial printed seven lines for roughly three disagreements: the same
    source and target repeated once per effect word. The pair is the
    consequential unit — `spill_1 → hazmat_worker_1` matters; whether the model
    called it `exposes` or `blocks_access_to` is a footnote, so the verbs go in
    brackets.

    Ordering uses the hazard-severity table (F34), so the line that matters is
    the first one read.
    """
    if not edges:
        return []
    from agentic.pathology import hazard_severity
    state_of = {}
    for g in ((s4.get("graph_a") or {}), (s4.get("graph_b") or {})):
        for n in (g.get("nodes") or []):
            if isinstance(n, dict) and n.get("id"):
                state_of.setdefault(str(n["id"]), str(n.get("state") or ""))
    label_of = {k: k.rsplit("_", 1)[0] for k in state_of}

    pairs: dict[tuple, list] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        src, tgt = str(e.get("source") or ""), str(e.get("target") or "")
        pairs.setdefault((src, tgt), []).append(str(e.get("effect") or ""))

    def _sev(src: str) -> float:
        v = hazard_severity(label_of.get(src, ""), state_of.get(src, ""))
        return -1.0 if v is None else v

    rows = [html.Div(f"{title} ({len(pairs)}):",
                     style={"fontSize": "11px", "color": "#b45309",
                            "fontWeight": "600", "marginTop": "6px"})]
    for (src, tgt), effects in sorted(pairs.items(),
                                      key=lambda kv: -_sev(kv[0][0])):
        verbs = ", ".join(sorted(set(e for e in effects if e)))
        sev = hazard_severity(label_of.get(src, ""), state_of.get(src, ""))
        sev_txt = f"   hazard severity {sev}" if sev is not None else ""
        rows.append(html.Div(
            [chipper(src), html.Span(" → "), chipper(tgt),
             html.Span(f"  ({verbs}){sev_txt}",
                       style={"color": "#94a3b8", "fontSize": "10.5px"})],
            style={"fontSize": "11.5px", "padding": "1px 0 1px 10px"}))
    return rows


def _runoff_rows(ro: dict) -> list:
    """JUDGES.md step 1 — the runoff judge's twin verdicts, rendered under the
    uncertainty panel whose candidates it judged.

    TWINS. The same judge model answers twice: text-only (the OFFICIAL
    verdict) and image-aware (the witness). Only the agreement is displayed —
    no routing, no arbitration — because what disagreement MEANS is
    deliberately undecided until live runs show how the twins actually split
    (Sunny, 2026-08-08). A disagreement chip names both verdicts so the reader
    can see the split, not just that one exists.
    """
    if not ro:
        return []
    label = {"answer_a": "candidate A", "answer_b": "candidate B",
             "equally_good": "both equally good", "unclear": "no clear verdict"}
    t = ro.get("text") or {}
    im = ro.get("image")
    rows = [html.Div([
        html.Span("RUNOFF · of the two leading candidates, which does the "
                  "record support?", className="unc-tag"),
        html.Span("  advisory, twin-judged", style={"fontSize": "10px",
                                                    "color": "#94a3b8"}),
    ], style={"marginTop": "5px"})]
    rows.append(html.Div([
        html.Span("text-only judge: ", style={"color": "#64748b"}),
        html.B(label.get(t.get("verdict"), t.get("verdict", "?"))),
        html.Span(f"  ({t.get('votes', 0)}/{t.get('n', 0)})",
                  style={"color": "#94a3b8"}),
    ], style={"fontSize": "11.5px", "padding": "1px 0 1px 8px"}))
    if im:
        rows.append(html.Div([
            html.Span("image-aware judge: ", style={"color": "#64748b"}),
            html.B(label.get(im.get("verdict"), im.get("verdict", "?"))),
            html.Span(f"  ({im.get('votes', 0)}/{im.get('n', 0)})",
                      style={"color": "#94a3b8"}),
        ], style={"fontSize": "11.5px", "padding": "1px 0 1px 8px"}))
        agree = ro.get("twins_agree")
        rows.append(html.Div(
            "twins agree" if agree else
            f"twins DISAGREE (text: {label.get(t.get('verdict'), '?')} · "
            f"image: {label.get(im.get('verdict'), '?')})",
            style={"fontSize": "11px", "fontWeight": "700",
                   "color": "#16a34a" if agree else "#b45309",
                   "padding": "1px 0 1px 8px"}))
    else:
        rows.append(html.Div("image twin did not run (no image available)",
                             style={"fontSize": "10px", "color": "#cbd5e1",
                                    "fontStyle": "italic",
                                    "padding": "0 0 0 8px"}))
    # the two candidates, one line each, so the verdict is readable in place
    for tag, cand in (("A", ro.get("candidate_a")),
                      ("B", ro.get("candidate_b"))):
        rows.append(html.Details([
            html.Summary(f"candidate {tag}", style={"fontSize": "10.5px",
                                                    "color": "#64748b",
                                                    "cursor": "pointer"}),
            html.Pre(str(cand or ""), style={"fontSize": "10px",
                                             "whiteSpace": "pre-wrap",
                                             "margin": "2px 0 4px 8px",
                                             "color": "#475569"}),
        ], style={"paddingLeft": "8px"}))
    facts = ro.get("code_facts") or {}
    inv = [x for k in ("invented_ids_a", "invented_ids_b")
           for x in (facts.get(k) or [])]
    if inv:
        rows.append(html.Div(
            f"code note: invented entity ids in play — {', '.join(sorted(set(inv)))}",
            style={"fontSize": "10.5px", "color": "#b45309",
                   "padding": "1px 0 1px 8px"}))
    return rows


def _graph_judge_rows(gj: dict) -> list:
    """F38 — the two questions the A-vs-B arithmetic cannot answer.

    ADVISORY, like the card judge, and labelled so. Q1's verdict is the
    evidence for "graph A is minimizing the victims" without the judge ever
    being asked to diagnose anything — the naming belongs to the pathology
    layer.
    """
    if not gj:
        return []
    rows: list[Any] = [html.Div("JUDGE · ADVISORY, NOT SCORED",
                                style={"fontSize": "11px", "color": "#64748b",
                                       "fontWeight": "600",
                                       "marginTop": "6px"})]
    # F53 follow-up: when Q1 sat out, code says WHY and states the
    # informative fact itself (e.g. Graph B's exposed set contains no
    # declared at-risk entity) — which needs no judge.
    note = gj.get("victims_note")
    if note:
        rows.append(html.Div("◈ " + str(note),
                             style={"fontSize": "11.5px", "color": "#b45309",
                                    "padding": "1px 0"}))
    v = gj.get("victims") or {}
    if v.get("n"):
        # F43: name BOTH sets and say which is worse. "Graph B's harmed
        # entities are more exposed" made the reader hold in their head that
        # Graph B is the model's independent belief, Graph A is the advice, and
        # that the point is the advice is protecting the wrong people.
        vic = v.get("sets") or {}
        who = {"graph_a": "Graph A's", "graph_b": "Graph B's"}.get(
            v.get("verdict"))
        # F39: this question has THREE answers, so random guessing already
        # gives the winner about 1.7 votes out of 5. A 3/5 is barely above
        # that — it reads as a verdict and is closer to a coin landing. 5/5 is
        # a verdict: guessing produces it about once in eighty tries. The card
        # judge already marked thin majorities; this one did not.
        n, votes = v.get("n") or 0, v.get("votes") or 0
        thin = bool(n) and votes <= (n // 2) + 1 and votes < n
        split = f"  ({votes}/{n})" + ("  · thin, close to a coin flip"
                                      if thin else "")
        if who:
            # F53 follow-up (Sunny: "less vague"). The old sentence packed
            # both sets and both roles into one clause and came out garbled
            # ("the entities the model protects are endangered are in MORE
            # danger than the ones the advice believes"). Three lines: what
            # each account says, then the verdict in plain words.
            a_set = ", ".join(vic.get("graph_a") or []) or "—"
            b_set = ", ".join(vic.get("graph_b") or []) or "—"
            rows.append(html.Div(
                f"the ADVICE acts as if endangered: {a_set}",
                style={"fontSize": "11px", "color": "#64748b",
                       "padding": "1px 0 0 10px"}))
            rows.append(html.Div(
                f"asked independently, the model BELIEVES endangered: {b_set}",
                style={"fontSize": "11px", "color": "#64748b",
                       "padding": "0 0 0 10px"}))
            if v.get("verdict") == "graph_b":
                meaning = ("◈ judge: the model's own belief names the set in "
                           "MORE danger — the advice may be protecting the "
                           "lesser set")
            else:
                meaning = ("◈ judge: the advice covers the more endangered "
                           "set — the model's own belief UNDERSTATES who is "
                           "in danger")
            rows.append(html.Div(
                meaning + split,
                style={"fontSize": "11.5px", "color": "#be123c",
                       "fontWeight": "600", "padding": "1px 0"}))
        elif v.get("verdict") == "equally":
            rows.append(html.Div(
                f"◆ both sets are equally exposed{split}",
                style={"fontSize": "11.5px", "color": "#7c3aed",
                       "padding": "1px 0"}))
        else:
            rows.append(html.Div(
                f"◇ the judge could not decide which set is more "
                f"exposed{split}",
                style={"fontSize": "11.5px", "color": "#94a3b8",
                       "padding": "1px 0"}))
    for m in (gj.get("mechanisms") or []):
        verdict = m.get("verdict")
        n, votes = m.get("n") or 0, m.get("votes") or 0
        split = f"  ({votes}/{n})" + (
            "  · thin" if (n and votes <= (n // 2) + 1 and votes < n) else "")
        same = verdict == "same_response"
        text = {"same_response": "same response either way — the difference is "
                                 "wording",
                "different_response": "a responder would DO something different",
                "unclear": "the judge could not decide"}.get(verdict, "")
        rows.append(html.Div(
            ("◆ " if same else "◇ " if verdict == "unclear" else "◈ ")
            + f"{m.get('source')} → {m.get('target')}: "
              f"'{m.get('effect_a')}' vs '{m.get('effect_b')}' — {text}{split}",
            style={"fontSize": "11.5px", "padding": "1px 0",
                   "color": "#7c3aed" if same else
                            ("#94a3b8" if verdict == "unclear" else "#be123c")}))
    return rows


def _trust_explanation_rows(trust: dict, alignment: dict) -> list:
    """F48. Three lines instead of a paragraph, and entity ids in line 2.

    The old explanation ran to four or five clauses, took the same shape on
    every scene, and never named a single entity — "2 seen-but-not-acted"
    where it could say "hazmat_worker_1, hazmat_worker_2". A reader deciding
    whether to act on emergency advice cannot do anything with a count.

    Falls back to the stored paragraph for runs recorded before F48, so old
    event streams still replay with something to read.
    """
    from agentic.errors4 import explain_trust
    try:
        lines = explain_trust(trust, alignment)
    except Exception:
        lines = []
    if not any(lines):
        return [html.Div(trust.get("explanation", ""),
                         style={"fontSize": "12px", "color": "#334155",
                                "margin": "2px 0 6px"})] if trust.get(
            "explanation") else []
    styles = [{"fontSize": "12.5px", "color": "#334155", "fontWeight": "600"},
              {"fontSize": "11.5px", "color": "#b45309"},
              {"fontSize": "11px", "color": "#64748b"}]
    return [html.Div(t, style={**st, "margin": "1px 0"})
            for t, st in zip(lines, styles) if t] + \
           [html.Div(style={"marginBottom": "5px"})]


def _singular_error_rows(trust: dict) -> list:
    """F48. The named errors, priced by consequence, shown as their own block.

    These do NOT come out of the weighted average — they are subtracted after
    it — so they need their own place on screen or the arithmetic looks wrong:
    a reader adding up the six contributors would not reach the score.
    """
    errs = [e for e in (trust.get("singular_errors") or []) if isinstance(e, dict)]
    if not errs:
        return []
    rows = [html.Div([
        html.Span("SERIOUS SINGLE ERRORS", className="unc-tag"),
        html.Span(f"  −{trust.get('singular_deduction', 0):.2f} off "
                  f"{trust.get('weighted_score', 0):.2f}",
                  title="priced by consequence: the same error costs more when "
                        "it happens to a person than to a vehicle. Subtracted "
                        "AFTER the weighted checks, not averaged in.",
                  style={"fontSize": "11px", "color": "#b91c1c",
                         "fontWeight": "700"})],
        style={"marginTop": "4px"})]
    for e in errs:
        rows.append(html.Div([
            html.Span(f"−{e.get('deduction', 0):.2f}  ",
                      style={"fontWeight": "700", "color": "#b91c1c"}),
            html.Span(str(e.get("detail", ""))),
            html.Span(f"  ({e.get('ceiling')} × {e.get('consequence')} "
                      f"consequence)", style={"color": "#94a3b8"}),
        ], style={"fontSize": "11.5px", "padding": "1px 0 1px 8px"}))
    return rows


def _effect_toggle_rows(al: dict) -> list:
    """F45. The effect switch, and the wiring check the default cannot see.

    a_fidelity now IGNORES the effect word by default — `exposes`, `may_harm`
    and `may_spread_to` all came out of the same model for the same tanker on
    the same scene, so counting them as three disagreements measured
    vocabulary rather than grounding.

    Two things still have to be reachable, so they live one click away rather
    than on screen:

      EFFECT COUNTED   the old whole-edge numbers. Every run before this
                       change is quoted in them, and a big gap between the
                       two settings is itself the finding — it means the
                       graphs agree on who endangers whom and disagree on
                       HOW, which is exactly what the graph judge's second
                       question is for.

      SAME PAIRS       who threatens whom, effect ignored. The default is a
                       mean of two SETS, and sets do not check wiring: name
                       the same hazards and the same victims, cross the wires
                       between them, and the default reads 1.00 while the two
                       graphs agree on no single claim. This number is the
                       one that would catch it.

    html.Details, no callback — the same idiom the rest of the page uses.
    """
    dc = al.get("decomposition") or {}
    a_s, b_s = al.get("a_fidelity_strict"), al.get("b_coverage_strict")
    pairs = dc.get("pairs")
    if a_s is None and pairs is None:
        return []

    body: list = []
    if pairs is not None:
        crossed = (dc.get("hazards") == 1.0 and dc.get("victims") == 1.0
                   and pairs < 1.0)
        body.append(html.Div(
            [html.Span(f"{pairs:.2f}  ", style={"fontWeight": "600",
                                                "color": "#334155"}),
             html.Span("same pairs"),
             html.Span("  — who threatens whom, still ignoring the effect "
                       "word", style={"color": "#94a3b8"})],
            style={"fontSize": "11.5px", "padding": "1px 0 1px 10px"}))
        if crossed:
            body.append(html.Div(
                "⚠ same hazards and same victims, but wired to each other "
                "differently — the two graphs agree on the cast and "
                "disagree on every claim",
                style={"fontSize": "11px", "color": "#b45309",
                       "fontWeight": "600", "padding": "1px 0 1px 10px"}))
    if a_s is not None:
        body.append(html.Div(
            [html.Span(f"{a_s:.2f} / {b_s:.2f}  ",
                       style={"fontWeight": "600", "color": "#334155"}),
             html.Span("a_fidelity / b_coverage with the effect word COUNTED"),
             html.Span("  — the pre-F45 definition; every earlier run is "
                       "quoted in these", style={"color": "#94a3b8"})],
            style={"fontSize": "11.5px", "padding": "1px 0 1px 10px"}))
    return [html.Details([
        html.Summary("▸ effect word: IGNORED  (click for effect-counted, "
                     "and the wiring check)",
                     style={"fontSize": "11px", "color": "#64748b",
                            "cursor": "pointer", "padding": "3px 0"}),
        html.Div(body),
    ])]


def _ab_decomposition_rows(dc: dict) -> list:
    """F35 — the A-vs-B comparison broken into the parts of an edge.

    a_fidelity counts WHOLE edges, so two graphs that name exactly the same
    hazards and disagree only about who those hazards threaten score 0.00 —
    identical to two graphs with nothing in common. On D_aerial the model
    agreed COMPLETELY about what the hazards were (1.00) and pointed them at
    the wrong people (victims 0.25), and the panel said 0.00.

    a_fidelity is unchanged and still leads. This sits under it so the same
    number can be read correctly.
    """
    if not dc or dc.get("hazards") is None:
        return []
    rows = [html.Div(dc.get("reading", ""),
                     style={"fontSize": "11.5px", "color": "#4f46e5",
                            "fontWeight": "600", "margin": "5px 0 2px"})]
    # F45: these two are no longer commentary BESIDE a_fidelity — they are the
    # two halves it is the mean of. The `+` and `=` make that readable without
    # a sentence explaining it.
    for key, glyph, label, hint in (
            ("hazards", "", "same hazards",
             "do both name the same sources of harm"),
            ("victims", "+", "same victims",
             "the same people or things harmed")):
        v = dc.get(key)
        if v is None:
            continue
        rows.append(html.Div(
            [html.Span(f"{glyph:<2}", style={"color": "#94a3b8"}),
             html.Span(f"{v:.2f}  ", style={"fontWeight": "600",
                                            "color": "#334155"}),
             html.Span(label),
             html.Span(f"  — {hint}", style={"color": "#94a3b8"})],
            style={"fontSize": "11.5px", "padding": "1px 0 1px 10px"}))
    # Name them. "0.25 of victims" sends you looking; "hazmat_worker_1 and
    # hazmat_worker_2" tells you who the advice left out.
    # F39: name them, both directions and both roles. "0.25 of victims" sends
    # you looking; the names tell you who the advice left out and who it
    # invented.
    for key, label, col in (
            ("hazards_only_in_b", "hazards the model believes in, no "
                                  "recommendation acts on", "#b45309"),
            ("hazards_only_in_a", "hazards the advice asserts, the model does "
                                  "not independently hold", "#be123c"),
            ("victims_only_in_b", "believed at risk, not acted on", "#b45309"),
            ("victims_only_in_a", "the advice protects, the model does not "
                                  "believe endangered", "#be123c")):
        who = dc.get(key) or []
        if who:
            rows.append(html.Div(f"{label}: {', '.join(who)}",
                                 style={"fontSize": "11.5px", "color": col,
                                        "padding": "1px 0 1px 10px"}))
    return rows


def _across_recommendations(s4: dict) -> list:
    """F29 — the panel below the cards: findings about the SET.

    A card footer disappears when its card is clean, and that absence means
    something. This panel does the OPPOSITE and always renders, because a
    missing panel would be ambiguous between "no cross-card problems" and "not
    rendered" — and the absence of duplication is a real positive signal, not
    the absence of a card.

    Three blocks, kept apart because they mean different things:

      COVERAGE   the set as a whole misses something. This is b_coverage in
                 words — "the dog is unaddressed" instead of "0.00".
      PAIRWISE   one card repeats another: the PADDING shape, count produced
                 in place of content.
      MODE       what the set acts on. Not a violation — a set with no
                 hazard-directed action is one the intervention gate cannot
                 test at all, which is the most important thing to know about
                 such a run and appeared nowhere before this.
    """
    rep = s4.get("set_report") or {}
    if not rep:
        # F37: runs that predate F29 have no set_report, and their set-level
        # findings — "at-risk X is not addressed by any recommendation" — used
        # to render in the INTERNAL ALIGNMENT panel that F37 removed. Without
        # this fallback they vanish from every older run, which is exactly the
        # coverage evidence the bias work needs.
        legacy = [f for f in ((s4.get("internal_alignment") or {})
                              .get("failures") or [])
                  if f.get("rank") is None]
        if not legacy:
            return []
        rep = {"coverage": legacy, "pairwise": [], "n_cards": 0,
               "modes": {}, "mode_verdict": "", "suppression_testable": 0}
    ia = (s4.get("internal_alignment") or {})
    conf = (s4.get("conformance") or {})
    card_conf = (conf.get("by_graph") or {}).get("card") or {}
    # F46. The score for the card rules was computed and never shown; the
    # header carried a bare tally ("card rule breaks 7"), which cannot be
    # compared between runs because it has no denominator on screen.
    #
    # It is NOT a pass fraction. It is 1 - severity/(severity + size): 1.00
    # means nothing broken, and it falls as breaks get more severe. So it is
    # labelled "clean", which says which way is good without a legend, and the
    # tooltip carries the definition for anyone who wants it.
    ea = (s4.get("explanation_alignment") or {})
    score_bits = f"  {rep.get('n_cards', 0)} cards"
    if card_conf:
        score_bits += f" · {card_conf.get('count', 0)} rule breaks"
    if ea.get("score") is not None:
        score_bits += f" · {ea['score']:.2f} clean"
    if ia.get("score") is not None:
        score_bits += f" · {ia['score']:.2f} they hang together"
    rows: list[Any] = [html.Div(
        [html.Span("ACROSS ALL RECOMMENDATIONS", className="unc-tag"),
         html.Span(score_bits, title=_CLEAN_TIP,
                   style={"fontSize": "11px", "color": "#94a3b8"})])]

    def block(title, note, findings, empty_text):
        rows.append(html.Div(f"{title} — {note}",
                             style={"fontSize": "11px", "color": "#64748b",
                                    "fontWeight": "600", "marginTop": "6px"}))
        if not findings:
            rows.append(html.Div(f"✓ {empty_text}",
                                 style={"fontSize": "11.5px",
                                        "color": "#16a34a", "padding": "1px 0"}))
            return
        for f in findings:
            sev = f.get("severity", 0)
            rows.append(html.Div(
                ("○ " if sev == 0 else "⚠ ") + str(f.get("detail", "")),
                style={"fontSize": "11.5px", "padding": "1px 0",
                       "color": "#94a3b8" if sev == 0 else "#b45309"}))

    block("COVERAGE", "what the set as a whole misses",
          rep.get("coverage") or [], "every at-risk entity is acted on")
    block("PAIRWISE", "one card repeating another",
          rep.get("pairwise") or [],
          "no duplicated quads, actions or remaining risks")

    m = rep.get("modes") or {}
    rows.append(html.Div("WHAT THE SET ACTS ON",
                         style={"fontSize": "11px", "color": "#64748b",
                                "fontWeight": "600", "marginTop": "6px"}))
    rows.append(html.Div(
        f"hazard-directed {m.get('hazard_directed', 0)} · "
        f"victim-directed {m.get('victim_directed', 0)} · "
        f"mixed {m.get('mixed', 0)} · "
        f"unattributed {m.get('unattributed', 0)}",
        style={"fontSize": "11.5px", "color": "#4f46e5", "padding": "1px 0"}))
    testable = rep.get("suppression_testable", 0)
    rows.append(html.Div(
        rep.get("mode_verdict", ""),
        style={"fontSize": "11.5px", "padding": "1px 0",
               "color": "#16a34a" if testable else "#be123c"}))

    # F30 — the same rollup the other blocks give conformance and alignment,
    # for the judge. ADVISORY: it moves no score, and the label says so, the
    # way the card footer already does.
    roll = (s4.get("card_judge") or {}).get("rollup") or {}
    if roll:
        rows.append(html.Div("SEMANTIC · ADVISORY, NOT SCORED",
                             style={"fontSize": "11px", "color": "#64748b",
                                    "fontWeight": "600", "marginTop": "6px"}))
        rows.append(html.Div(roll.get("headline", ""),
                             style={"fontSize": "11.5px", "color": "#7c3aed",
                                    "padding": "1px 0"}))
        for f in roll.get("findings") or []:
            # ◈ the judge found something · ◇ the judge could not decide.
            # Kept apart on purpose: an undecided verdict is a finding about
            # the INSTRUMENT, and dressing it up as a defect of the model is
            # exactly the mistake F26 and F28 were.
            undecided = f.get("kind") == "undecided"
            votes = (f"  ({f.get('votes')}/{f.get('n')})"
                     if f.get("n") else "")
            thin = "  · thin majority" if f.get("thin") else ""
            rows.append(html.Div(
                ("◇ " if undecided else "◈ ") + str(f.get("text", ""))
                + votes + thin,
                style={"fontSize": "11.5px", "padding": "1px 0",
                       "color": "#94a3b8" if undecided else "#be123c"}))
    return [html.Div(rows, className="unc-panel")]


def _card_verdict_rows(findings: list, mode: str, judged: dict | None):
    """F24 — one recommendation card's own verdict, rendered under it.

    Three bands, kept visibly apart because they answer different questions and
    pathology has to tell them apart:

      conformance   this surface broke a rule                (charged)
      alignment     two surfaces disagree with each other   (charged)
      judge         is the explanation hollow               (ADVISORY, never scored)

    Severity-0 findings are shown in grey, not amber: they are the cases where
    OUR constraint was unsatisfiable — recorded because nothing is erased, but
    not the model's error and never charged. Showing them in the same colour as
    a real defect would undo the point of separating them.
    """
    MODE_TEXT = {
        "hazard_directed": "acts on the hazard · testable by suppression",
        "victim_directed": "acts on the victim · needs its own intervention",
        "mixed": "acts on both sides",
        "unattributed": "acts on neither side of the claim",
    }
    rows: list[Any] = []
    if mode:
        rows.append(html.Div(
            f"◆ {mode.replace('_', '-')} — {MODE_TEXT.get(mode, '')}",
            style={"fontSize": "11px", "color": "#4f46e5", "marginTop": "5px"}))

    card = [f for f in findings if f.get("level") != "set"]
    for signal, label, colour in (
            ("conformance", "conformance", "#b45309"),
            ("internal_alignment", "alignment", "#be123c")):
        mine = [f for f in card if f.get("signal") == signal]
        if not mine:
            continue
        rows.append(html.Div(label.upper(), className="unc-tag",
                             style={"marginTop": "5px"}))
        for f in sorted(mine, key=lambda x: -x.get("severity", 0)):
            sev = f.get("severity", 0)
            rows.append(html.Div(
                [html.Span("○ " if sev == 0 else "⚠ "),
                 html.Span(f.get("detail", ""))],
                style={"fontSize": "11.5px", "padding": "1px 0",
                       "color": "#94a3b8" if sev == 0 else colour}))
            if sev == 0:
                rows.append(html.Div(
                    "recorded, not charged — no legal answer was available",
                    style={"fontSize": "10px", "color": "#cbd5e1",
                           "fontStyle": "italic", "marginLeft": "14px"}))

    if judged:
        # The VOTE SPLIT rides beside every verdict, not just the winner.
        # "aligned 5/5" and "aligned 3/5" are different findings, and a split
        # is the judge saying it cannot see the boundary — worth more than the
        # verdict it happened to land on.
        rows.append(html.Div("JUDGE · ADVISORY, NOT SCORED",
                             className="unc-tag", style={"marginTop": "5px"}))
        for surface, label in (("prose", "reason"), ("structure", "quad")):
            v = judged.get(surface) or {}
            verdict = v.get("verdict", "unclear")
            ok = verdict == "causally_aligned"
            mark = "◆" if ok else ("◇" if verdict == "unclear" else "◈")
            text = {"causally_aligned": "causally aligned with the action",
                    "not_causally_aligned":
                        "NOT causally aligned — the action would not reduce "
                        "the harm this describes",
                    "unclear": "the judge could not decide"}.get(verdict, "")
            split = (f"  ({v.get('votes', 0)}/{v.get('n', 0)})"
                     if v.get("n") else "")
            rows.append(html.Div(
                f"{mark} {label}: {text}{split}",
                style={"fontSize": "11.5px", "padding": "1px 0",
                       "color": "#7c3aed" if ok else "#be123c"}))
        sc = judged.get("same_claim") or {}
        if sc.get("n"):
            same = sc.get("verdict") == "yes"
            rows.append(html.Div(
                ("◆ reason and quad make the same claim" if same else
                 ("◇ the judge could not say whether reason and quad agree"
                  if sc.get("verdict") == "unclear" else
                  "◈ reason and quad make DIFFERENT claims"))
                + f"  ({sc.get('votes', 0)}/{sc.get('n')})",
                style={"fontSize": "11.5px", "padding": "1px 0",
                       "color": "#7c3aed" if same else "#be123c"}))
    if not rows:
        return []
    return [html.Div(rows, style={"borderTop": "1px dashed #e2e8f0",
                                  "marginTop": "6px", "paddingTop": "4px"})]


def _rec_uncertainty_row(threat: str, unc: dict, chipper):
    """The granular measured-uncertainty slice for ONE recommendation, joined
    by its threat: did this threat reappear across the re-asks, and did its
    causal mechanism hold? Returns None when there's nothing notable to show."""
    threat = str(threat or "").split("·")[0].strip()
    n = int((unc or {}).get("n_probes") or 0)
    if not n or not threat:
        return None
    gran = (unc or {}).get("granular") or {}
    tu = (gran.get("threats") or {}).get(threat)
    eu = (gran.get("effects") or {}).get(threat)
    bits: list[Any] = []
    if tu is None:
        bits.append(html.Span(f"U 1.0 · never reappeared in {n} re-asks",
                              style={"color": "#b91c1c", "fontWeight": "600"}))
    elif tu.get("u", 0) > 0:
        bits.append(html.Span(f"U {tu.get('u')} · named a threat in only "
                              f"{tu.get('votes')} re-asks",
                              style={"color": "#b45309"}))
    if eu and eu.get("u", 0) > 0:
        if bits:
            bits.append(html.Span(" · ", style={"color": "#cbd5e1"}))
        bits.append(html.Span(f"U {eu.get('u')} · mechanism splits: "
                              f"{eu.get('evidence')}", style={"color": "#b45309"}))
    if not bits:
        return None
    return html.Div(
        [html.Span("uncertainty: ", style={"color": "#64748b",
                                           "fontSize": "11px"}),
         chipper(threat), html.Span(" ")] + bits,
        style={"fontSize": "11px", "marginTop": "3px", "paddingTop": "3px",
               "borderTop": "1px dashed #e2e8f0"})


def _votes_to_chips(evidence: str, chipper) -> list[Any]:
    """Render a votes string like 'house_1×3, dust_1×2' as role-colored chips:
    [chip(house_1) ×3 · chip(dust_1) ×2]. A token that isn't 'entity×n' falls
    back to plain text, so it never breaks on '∅' or odd input."""
    out: list[Any] = []
    for i, tok in enumerate(str(evidence or "").split(", ")):
        if i:
            out.append(html.Span(" · ", style={"color": "#cbd5e1"}))
        ent, sep, cnt = tok.partition("×")
        ent = ent.strip()
        if sep and ent and ent != "∅":
            out.append(chipper(ent))
            out.append(html.Span(f"×{cnt}", style={"fontSize": "10.5px",
                                                   "color": "#94a3b8"}))
        else:
            out.append(html.Span(tok, style={"fontSize": "10.5px",
                                             "color": "#94a3b8"}))
    return out


def _make_chipper(graph: dict[str, Any]):
    """Return chip(object_id) → a role-colored entity chip, using a graph's
    node roles. at-risk is derived from structure (target of a harm edge) so
    it works even for Graph B (whose nodes carry no at_risk flag). Ids not in
    the graph render gray with the literal text (e.g. an un-normalized pick).
    A trailing '·state' on the id is stripped for the lookup."""
    nodes = {n.get("id"): n for n in (graph.get("nodes") or [])
             if isinstance(n, dict)}
    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
    # A non-hazard entity that a hazard THREATENS is at-risk — whether the
    # harm reaches it directly (may_harm), by the hazard spreading to it
    # (may_spread_to), or by exposure/isolation/blocked access. This keeps a
    # quad's affected_objects colored consistently regardless of the effect.
    # (increases_risk_to / worsens target hazards or latent threats, not
    # victims, so they are left to their own role.)
    harm = {"may_harm", "threatens", "may_spread_to",
            "exposes", "isolates", "blocks_access_to"}
    at_risk_ids = {nid for nid, n in nodes.items() if n.get("at_risk")}
    for e in edges:
        tgt = str(e.get("target", ""))
        if str(e.get("effect", "")) in harm and not nodes.get(tgt, {}).get(
                "hazardous"):
            at_risk_ids.add(tgt)

    def chip(oid: str):
        oid = str(oid)
        key = oid.split("·")[0].strip()          # tolerate "house_1·burning"
        return _node_chip(key, nodes.get(key, {}), at_risk=key in at_risk_ids)
    return chip


def causal_graph_view(graph: dict[str, Any], title: str,
                      open_default: bool = False) -> html.Details:
    """A collapsible causal-graph view that shows EVERY node and edge and
    still fits the narrow panel. Collapsed by default (just the counts) so
    the panel isn't crowded; expand to see: edges grouped by source hazard
    (targets as role-colored chips), then a chip row of every node."""
    from collections import OrderedDict
    nodes = {n.get("id"): n for n in (graph.get("nodes") or [])
             if isinstance(n, dict)}
    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]

    chip = _make_chipper(graph)

    summary = html.Summary([
        html.Span(title, className="unc-tag"),
        html.Span(f"  {len(nodes)} nodes · {len(edges)} edges",
                  style={"fontSize": "11px", "color": "#64748b"}),
        html.Span(" ▾", className="station-chev"),
    ], style={"cursor": "pointer", "listStyle": "none"})

    body: list[Any] = []

    # EDGES — grouped by source hazard; targets as chips
    by_source: "OrderedDict[str, list]" = OrderedDict()
    for e in edges:
        by_source.setdefault(str(e.get("source", "")), []).append(e)
    if by_source:
        body.append(html.Div("EDGES", style={"fontSize": "10px",
                                             "letterSpacing": ".5px",
                                             "color": "#94a3b8",
                                             "margin": "6px 0 2px"}))
    for src, es in by_source.items():
        by_effect: "OrderedDict[str, list]" = OrderedDict()
        for e in es:
            by_effect.setdefault(str(e.get("effect", "")), []).append(
                str(e.get("target", "")))
        eff_rows = []
        for effect, targets in by_effect.items():
            eff_rows.append(html.Div(
                [html.Span(f"—{effect}→ ", style={"color": "#7c3aed",
                                                  "fontSize": "11.5px"})]
                + [chip(t) for t in targets],
                style={"paddingLeft": "12px", "lineHeight": "1.7"}))
        body.append(html.Div([chip(src), html.Div(eff_rows)],
                             style={"margin": "4px 0"}))
    if not edges:
        body.append(html.Div("no edges", style={"fontSize": "11px",
                                                "color": "#94a3b8"}))

    # F18 — ONE-ENDED CLAIMS, drawn as stub rows (Sunny's option A). The stub
    # occupies the space the real edge would occupy and points at an empty
    # endpoint, so the DIRECTION of the missing half is visible: a hazard
    # aimed at nobody reads differently from a victim with no source. Amber +
    # dashed + open endpoint so it can never be mistaken for a claim — and
    # amber, not grey, because grey dashing already means vlm_sam_fallback.
    #
    # These are rendered FROM THE NODE FLAGS and are never in edges[]; the
    # frozen comparators still see the real edge count.
    _stub = {"display": "inline-block", "width": "34px",
             "borderTop": "2px dashed #d97706", "verticalAlign": "middle",
             "margin": "0 6px"}
    _empty = {"display": "inline-block", "width": "11px", "height": "11px",
              "borderRadius": "50%", "border": "1.5px dashed #d97706",
              "verticalAlign": "middle"}
    _note = {"fontSize": "10px", "color": "#b45309", "marginLeft": "8px",
             "fontStyle": "italic"}
    one_ended = [n for n in nodes.values()
                 if n.get("unattached") or n.get("unattributed")]
    if one_ended:
        body.append(html.Div("ONE-ENDED CLAIMS",
                             style={"fontSize": "10px", "letterSpacing": ".5px",
                                    "color": "#b45309", "margin": "8px 0 2px"}))
    for n in one_ended:
        nid = str(n.get("id") or "")
        if n.get("unattached"):          # hazard -> nothing
            row = [chip(nid), html.Span(style=_stub), html.Span(style=_empty),
                   html.Span("no target named", style=_note)]
        else:                            # nothing -> victim
            row = [html.Span(style=_empty), html.Span(style=_stub), chip(nid),
                   html.Span("no source named", style=_note)]
        body.append(html.Div(row, style={"margin": "3px 0",
                                         "lineHeight": "1.9"}))

    # NODES — every node as a chip, so isolated ones are visible too
    body.append(html.Div("ALL NODES", style={"fontSize": "10px",
                                            "letterSpacing": ".5px",
                                            "color": "#94a3b8",
                                            "margin": "8px 0 2px"}))
    body.append(html.Div([chip(nid) for nid in nodes],
                        style={"lineHeight": "1.9"}))
    body.append(html.Div("red = hazard · amber = at-risk · gray = neutral",
                        style={"fontSize": "10px", "color": "#cbd5e1",
                               "marginTop": "4px"}))

    return html.Details([summary, html.Div(body)], open=open_default,
                        className="unc-panel")


def _entity_boxes(d: dict[str, Any]) -> dict[str, list]:
    """object_id -> a VALID bbox, from the final perception result, falling back
    to the bound box then the anchor. Same sources scene_component draws."""
    out: dict[str, list] = {}
    for o in (d.get("result") or {}).get("detected_objects", []) or []:
        if _valid_box(o.get("bbox")):
            out[o["object_id"]] = o["bbox"]
    for oid, bnd in (d.get("bound") or {}).items():
        if oid not in out and _valid_box((bnd or {}).get("bbox")):
            out[oid] = bnd["bbox"]
    for a in d.get("anchors") or []:
        oid = a.get("object_id")
        if oid and oid not in out and _valid_box(a.get("anchor_bbox")):
            out[oid] = a["anchor_bbox"]
    return out


def _hazard_unc_flag(oid: str, unc: dict) -> str | None:
    """Short uncertainty flag for a hazard, from the probes: does it flicker as
    a threat, or does its mechanism refuse to commit? None if stable."""
    n = int((unc or {}).get("n_probes") or 0)
    if not n:
        return None
    g = (unc or {}).get("granular") or {}
    tu = (g.get("threats") or {}).get(oid)
    eu = (g.get("effects") or {}).get(oid)
    if tu is None:
        return f"never reappeared in {n} re-asks"
    if tu.get("u", 0) > 0:
        return f"a threat in only {tu.get('votes')}"
    if eu and eu.get("u", 0) > 0:
        return "mechanism won't commit"
    return None


def _stage4_pins(d: dict[str, Any], chipper) -> list[Any]:
    """ON THE IMAGE: clickable labels pinned near each hazard and at-risk victim,
    rendered ONTO the main scene canvas (not a separate image). A label's compact
    form shows the role + consequence + an uncertainty flag; clicking it
    (html.Details, no callback) expands the recommendation that targets/protects
    that entity. Pure function of the frozen boxes + the Stage-4 result, so a
    reflection revision re-renders it — same as the perception/assessment boxes.
    Returns [] when there's nothing to pin."""
    s4 = d.get("stage4") or {}
    size = d.get("image_size")
    boxes = _entity_boxes(d)
    if not (size and boxes and s4):
        return []
    ga = s4.get("graph_a", {}) or {}
    nodes = {n.get("id"): n for n in ga.get("nodes", []) if isinstance(n, dict)}
    recs = s4.get("recommendations", []) or []
    unc = s4.get("uncertainty", {}) or {}
    per_rec = {p.get("rank"): p for p in
               ((s4.get("trust") or {}).get("per_rec") or [])}

    def _bare(x):
        return str(x or "").split("·")[0].strip()

    # threat -> the recommendations that target it (top consequence first)
    by_threat: dict[str, list] = {}
    victims: dict[str, list] = {}       # victim id -> protecting recs
    for r in recs:
        q = r.get("structured_reasoning", {}) or {}
        t = _bare(q.get("threat"))
        if t:
            by_threat.setdefault(t, []).append(r)
        for v in (q.get("affected_objects") or []):
            victims.setdefault(_bare(v), []).append(r)

    def _pin(oid, role, border, bg, summary_kids, detail_kids):
        style = {"position": "absolute", "maxWidth": "44%", "zIndex": 40,
                 # nudge below the perception box-tag so they don't collide
                 "transform": "translateY(16px)"}
        pc = _pct_box(boxes[oid], size)
        style["left"], style["top"] = pc["left"], pc["top"]
        return html.Details([
            html.Summary(summary_kids, style={
                "cursor": "pointer", "listStyle": "none", "fontSize": "10.5px",
                "fontWeight": "700", "padding": "2px 6px", "borderRadius": "6px",
                "background": bg, "border": f"1.5px solid {border}",
                "color": border, "whiteSpace": "nowrap", "boxShadow":
                "0 1px 4px #0003"}),
            html.Div(detail_kids, style={
                "fontSize": "10.5px", "background": "#fff",
                "border": f"1px solid {border}", "borderRadius": "6px",
                "padding": "4px 6px", "marginTop": "2px", "maxWidth": "220px",
                "boxShadow": "0 2px 8px #0002"}),
        ], style=style, className="s4-map-pin")

    labels: list[Any] = []
    placed: set[str] = set()

    # hazard pins (red) — the recommendation that targets each hazard
    for oid in boxes:
        rlist = by_threat.get(oid)
        is_haz = nodes.get(oid, {}).get("hazardous")
        if not (rlist or is_haz):
            continue
        rlist = sorted(rlist or [],
                       key=lambda r: -((per_rec.get(r.get("rank"), {})
                                        .get("consequence") or 0)))
        top = rlist[0] if rlist else None
        pr = per_rec.get(top.get("rank")) if top else {}
        flag = _hazard_unc_flag(oid, unc)
        cons = (pr or {}).get("consequence_band")
        summ = [chipper(oid)]
        if cons:
            summ.append(html.Span(f" ⚠{cons}", style={"color": "#b91c1c"}))
        if flag:
            summ.append(html.Span(" · U!", style={"color": "#b45309"}))
        detail: list[Any] = []
        if top:
            q = top.get("structured_reasoning", {}) or {}
            detail.append(html.Div([html.B(f"#{top.get('rank')} "),
                                    top.get("action", "")]))
            detail.append(html.Div(["harm: ", chipper(oid),
                                    html.Span(f" —{q.get('effect','')}→ ",
                                              style={"color": "#7c3aed"})]
                                   + [chipper(v) for v in
                                      (q.get("affected_objects") or [])],
                                   style={"marginTop": "2px"}))
            if pr:
                detail.append(html.Div(
                    f"trust {pr.get('score')} · consequence "
                    f"{pr.get('consequence_band')}", style={"marginTop": "2px",
                                                            "color": "#64748b"}))
        if flag:
            detail.append(html.Div(f"uncertainty: {flag}",
                                   style={"color": "#b45309", "marginTop": "2px"}))
        labels.append(_pin(oid, "hazard", "#b91c1c", "#fee2e2", summ, detail))
        placed.add(oid)

    # victim pins (amber) — who's at risk, and the rec that protects them
    for oid, rlist in victims.items():
        if oid in placed or oid not in boxes:
            continue
        if nodes.get(oid, {}).get("hazardous"):
            continue
        prot = sorted(rlist, key=lambda r: r.get("rank") or 99)[0]
        threat = _bare((prot.get("structured_reasoning", {}) or {}).get("threat"))
        summ = [chipper(oid), html.Span(" at risk", style={"color": "#b45309"})]
        detail = [html.Div(["threatened by ", chipper(threat)]),
                  html.Div([html.B(f"#{prot.get('rank')} "),
                            prot.get("action", "")], style={"marginTop": "2px"})]
        labels.append(_pin(oid, "victim", "#b45309", "#fef3c7", summ, detail))
        placed.add(oid)

    return labels




_RO_LAB = {"answer_a": "candidate A", "answer_b": "candidate B",
           "equally_good": "equal", "unclear": "unclear"}


def _bench_pointer(text: str) -> Any:
    """The one-line pointer left behind where a judge verdict used to render.
    The verdict itself lives on the bench (section 5) — one verdict, one
    home; the panel keeps just enough to know a judge spoke and where."""
    return html.Div("⚖ " + text + "  → see THE JUDGES' BENCH",
                    style={"fontSize": "10px", "color": "#7c3aed",
                           "fontWeight": "600", "marginTop": "3px"})


def _bench_card(title: str, task: str, verdict_rows: list,
                twin_chip: Any = None, foot: str = "",
                wide: bool = False) -> Any:
    """One judge, one card — Stage 2's TASK / AUTHORITY / THIS RUN grammar,
    purple frame, plus the twin-agreement chip. Sunny: judges were almost
    hidden inside the panels they judged; a judge must be recognizable as a
    judge and findable in exactly one place."""
    rows = [
        html.Div("⚖ " + title, style={
            "fontSize": "10px", "fontWeight": "800", "letterSpacing": ".08em",
            "color": "#fff", "background": "#7c3aed",
            "padding": "3px 10px", "borderRadius": "8px 8px 0 0"}),
        html.Div([html.Span("TASK ", className="unc-tag"),
                  html.Span(task, style={"fontSize": "10px"})],
                 style={"padding": "3px 8px 0"}),
        html.Div([html.Span("AUTHORITY ", className="unc-tag"),
                  html.Span("advises — never enters a score",
                            style={"fontSize": "10px",
                                   "fontStyle": "italic"})],
                 style={"padding": "0 8px"}),
        html.Div([html.Span("THIS RUN ", className="unc-tag")]
                 + verdict_rows, style={"padding": "0 8px 4px"}),
    ]
    if twin_chip is not None:
        rows.append(html.Div(twin_chip, style={"padding": "0 8px 4px"}))
    if foot:
        rows.append(html.Div(foot, style={"fontSize": "9.5px",
                                          "color": "#94a3b8",
                                          "padding": "0 8px 6px"}))
    return html.Div(rows, style={
        "border": "1px solid #7c3aed66", "borderRadius": "8px",
        "background": "#fff", "width": "420px" if wide else "246px",
        "verticalAlign": "top",
        "display": "inline-block", "margin": "0 8px 8px 0"})


def _twin_chip(ro: dict) -> Any:
    lab = {"answer_a": "candidate A", "answer_b": "candidate B",
           "equally_good": "equal", "unclear": "unclear"}
    t, im = ro.get("text") or {}, ro.get("image")
    if not im:
        return html.Span("image twin did not run", style={
            "fontSize": "9.5px", "color": "#cbd5e1", "fontStyle": "italic"})
    agree = ro.get("twins_agree")
    return html.Span(
        "✓ TWINS AGREE" if agree else
        f"⚠ TWINS DISAGREE (text: {lab.get(t.get('verdict'), '?')} · "
        f"image: {lab.get(im.get('verdict'), '?')})",
        style={"fontSize": "9.5px", "fontWeight": "800",
               "color": "#16a34a" if agree else "#b45309"})


def _runoff_bench_card(ro: dict, which: str) -> Any:
    lab = {"answer_a": "candidate A", "answer_b": "candidate B",
           "equally_good": "both equally good", "unclear": "no clear verdict"}
    t, im = ro.get("text") or {}, ro.get("image")

    def _v(name, v):
        return html.Div([html.Span(name + ": ", style={"color": "#64748b"}),
                         html.B(lab.get(v.get("verdict"), "?")),
                         html.Span(f" ({v.get('votes', 0)}/{v.get('n', 0)})",
                                   style={"color": "#94a3b8"})],
                        style={"fontSize": "10.5px"})
    rows = [_v("text-only", t)] + ([_v("image-aware", im)] if im else [])
    facts = ro.get("code_facts") or {}
    inv = sorted({x for k in ("invented_ids_a", "invented_ids_b")
                  for x in (facts.get(k) or [])})
    rows.append(html.Details([
        html.Summary("the two candidates", style={"fontSize": "9.5px",
                                                  "color": "#64748b",
                                                  "cursor": "pointer"}),
        html.Pre(f"A:\n{ro.get('candidate_a', '')}\n\nB:\n"
                 f"{ro.get('candidate_b', '')}",
                 style={"fontSize": "9px", "whiteSpace": "pre-wrap",
                        "color": "#475569", "margin": "2px 0"})]))
    if inv:
        rows.append(html.Div(f"code note: invented ids — {', '.join(inv)}",
                             style={"fontSize": "9.5px", "color": "#b45309"}))
    # F52: the judge's reasoning, readable in place. F51's loophole was found
    # by reading the stored reasoning off disk — the person at the screen had
    # no way to see it. Every vote, verdict-tagged, one click away.
    for name, tw in (("text-only", t), ("image-aware", im)):
        votes = (tw or {}).get("all_reasoning") or []
        if votes:
            rows.append(html.Details([
                html.Summary(f"{name} judge's reasoning ({len(votes)} votes)",
                             style={"fontSize": "9.5px", "color": "#7c3aed",
                                    "cursor": "pointer"}),
                html.Div([html.Div([
                    html.B(lab.get(v.get("verdict"), v.get("verdict", "?")),
                           style={"fontSize": "9px"}),
                    html.Pre(v.get("text", ""), style={
                        "fontSize": "9px", "whiteSpace": "pre-wrap",
                        "color": "#475569", "margin": "1px 0 6px"})])
                    for v in votes])]))
    # The TASK must say WHAT the candidates are and what "support" means —
    # "of the two leading probe candidates, which does the record support?"
    # told the reader a choice happened without saying what was chosen
    # between (Sunny). Each application states its own object and criterion,
    # mirroring the judge's actual prompt (COVERS / SUPPORTED, JUDGES.md).
    task = {
        "RECOMMENDATIONS": (
            "asked for recommendations {n} times, the model gave differing "
            "answers. Candidates = two complete recommendation SETS (each "
            "action + reason + causal claim). Verdict = which set better "
            "covers the declared dangers and at-risk entities, naming only "
            "entities that exist."),
        "GRAPH B": (
            "asked to draw its causal graph {n} times, the model gave "
            "differing answers. Candidates = two complete who-endangers-whom "
            "graphs. Verdict = which graph the scene record supports: real "
            "entities only, arrows out of dangerous states, every danger "
            "covered."),
    }[which].format(n=(ro.get("code_facts") or {}).get("n_asks", "several"))
    return _bench_card(f"RUNOFF · {which}", task, rows, _twin_chip(ro))


def _judges_bench(s4: dict, d: dict) -> list:
    """Section 5 — every subjective verdict of the run, one card per judge.

    The bench OWNS the verdicts; the panels that were judged keep one-line
    pointer chips. One verdict, one home (the F37 no-double-print rule,
    applied to judges)."""
    cards: list = []
    cj = s4.get("card_judge") or {}
    for v in (cj.get("verdicts") or []):
        judged_rows = _card_verdict_rows([], "", v)
        if judged_rows:
            cards.append(_bench_card(
                f"CARD JUDGE · rec {v.get('rank')}",
                "does each explanation actually explain the action?",
                judged_rows, None,
                "text-only (twin arrives with the A-vs-B judge build)",
                wide=True))
            continue
    gj = s4.get("graph_judge") or {}
    if gj:
        # F43's decisions travel with the verdict: the original renderer
        # names both victim sets, keeps the judge's words diagnosis-free,
        # and marks unclear verdicts apart. The bench reuses it verbatim.
        rows = _graph_judge_rows(gj)
        if rows:
            cards.append(_bench_card(
                "GRAPH JUDGE",
                "the two questions the A-vs-B arithmetic cannot answer",
                rows, None,
                "text-only (twin arrives with the A-vs-B judge build)",
                wide=True))
    ro = s4.get("runoff_judge") or {}
    if ro.get("recommendations"):
        cards.append(_runoff_bench_card(ro["recommendations"],
                                        "RECOMMENDATIONS"))
    if ro.get("graph_b"):
        cards.append(_runoff_bench_card(ro["graph_b"], "GRAPH B"))
    if not cards:
        return [html.Div("no judge ran on this run",
                         className="ticket-empty")]
    # bench header: twins + judge time, when the live event stream carried it
    twins = [x.get("twins_agree") for x in ro.values()
             if isinstance(x, dict) and x.get("twins_agree") is not None]
    head_bits = []
    if twins:
        head_bits.append(f"twins: {sum(1 for t in twins if t)} agree / "
                         f"{sum(1 for t in twins if not t)} disagree")
    jt = (d.get("s4_judge_seconds") or 0)
    if jt:
        head_bits.append(f"judge time {jt / 60:.0f} min")
    head = html.Div(
        ("  ·  ".join(head_bits) + "  ·  " if head_bits else "")
        + "every verdict on this bench is advisory — none moves a score",
        style={"fontSize": "10.5px", "color": "#64748b",
               "margin": "0 0 6px 2px"})
    return [head, html.Div(cards)]


_S4_SECTIONS = (
    ("verdict", "1 · THE VERDICT", "read this first", True),
    ("recs", "2 · THE RECOMMENDATIONS", "what the model said", True),
    ("picks", None, None, None),           # folded into section 2
    ("stability", "3 · STABILITY", "does it say the same thing twice?", False),
    ("graphs", "4 · THE CAUSAL GRAPHS", "the claim being tested", False),
    # Sunny (C run on the new prompt): A-vs-B is too important to bury inside
    # the graphs section — it is the earned comparison the trust factors read.
    ("alignment", "5 · ALIGNMENT",
     "does the advice match the model's own belief?", True),
    ("bench", "6 · THE JUDGES' BENCH",
     "all subjective, all advisory, one card each", True),
)


def _assemble_stage4_sections(out: list, s4: dict, d: dict) -> list:
    """Partition the built panels by the «sec:name» markers and emit five
    numbered, collapsible sections (Sunny, 2026-08-08: 'stage 4 is getting
    crowded'). The build ORDER above is unchanged — only the display order
    is decided here, so the marker scheme adds no new render logic."""
    buckets: dict[str, list] = {}
    cur = "verdict"
    for item in out:
        if isinstance(item, str) and item.startswith("«sec:"):
            cur = item[5:-1]
            continue
        buckets.setdefault(cur, []).append(item)
    # picks fold into the recommendations section, after the cards
    recs = buckets.get("recs", []) + buckets.get("picks", [])
    body = {"verdict": buckets.get("verdict", []), "recs": recs,
            "stability": buckets.get("stability", []),
            "graphs": buckets.get("graphs", []),
            "alignment": buckets.get("alignment", []),
            "bench": _judges_bench(s4, d)}
    sections: list = []
    for key, title, sub, open_ in _S4_SECTIONS:
        if title is None or not body.get(key):
            continue
        sections.append(html.Details([
            html.Summary([
                html.Span(title, style={"fontSize": "12px",
                                        "fontWeight": "800",
                                        "letterSpacing": ".04em"}),
                html.Span("  — " + sub, style={"fontSize": "10.5px",
                                               "color": "#94a3b8"}),
            ], style={"cursor": "pointer", "padding": "6px 8px",
                      "background": "#f1f5f9", "borderRadius": "6px"}),
            html.Div(body[key], style={"padding": "6px 0 2px 4px"}),
        ], open=open_, style={"marginBottom": "8px"}))
    return sections


def stage4_component(d: dict[str, Any], image_src: str | None = None) -> list[Any]:
    """STAGE 4 body: the ON-THE-IMAGE overlay, the trust headline, the three-way
    intervention picks, measured uncertainty, the checks, the model's
    recommendations with their quads + per-rec uncertainty, and both graphs."""
    err = d.get("stage4_error")
    s4 = d.get("stage4")
    if err and not s4:
        return [html.Div(f"Stage 4 could not run: {err}",
                         className="ticket-empty", style={"color": "#b45309"})]
    if not s4:
        # live running view: show the steps ticking as events arrive, so the
        # (slow, 5-probe) run never looks frozen. Blank only before it starts.
        marks = d.get("stage4_marks") or set()
        if not (d.get("stage4_started") or marks):
            return [html.Div("Stage 4 runs after assessment finishes…",
                             className="ticket-empty")]
        probe = d.get("stage4_probe", 0)
        LIVE = [("recommend", "recommend", "writing the plan"),
                ("uncertainty", "uncertainty",
                 f"re-asking to measure uncertainty · probe {probe}/5"),
                ("graph_a", "graph_a", "assembling Graph A"),
                ("graph_b", "graph_b", "asking for the independent Graph B"),
                ("picks", "picks", "choosing what to intervene on"),
                # the judges own their time on screen: they are the slow part
                # (gemma reasons on every vote), and without these rows their
                # ~20 minutes rendered as "folding the trust score" (Sunny).
                ("card_judge", "card_judge",
                 "the card judge is voting (minutes — it reasons per vote)"),
                ("graph_judge", "graph_judge",
                 "the graph judge is voting (minutes)"),
                ("runoff", "runoff",
                 f"the runoff twins are voting · "
                 f"{d.get('stage4_runoff', 0)}/2 comparisons done "
                 f"(the slowest step — text + image, per vote)"),
                ("trust", "trust", "folding the trust score")]
        rows: list[Any] = [html.Div("STAGE 4 · running…", className="unc-tag")]
        active_used = False
        for key, _k, label in LIVE:
            if key in marks:
                mark, col = "✓", "#16a34a"
            elif not active_used:
                mark, col, active_used = "⟳", "#2563eb", True
            else:
                mark, col = "○", "#cbd5e1"
            rows.append(html.Div(f"{mark} {label}",
                                 style={"fontSize": "12px", "color": col,
                                        "padding": "1px 0"}))
        return [html.Div(rows, className="unc-panel")]
    out: list[Any] = []

    # ── what to intervene on: the three picks + agreement ──
    # one chipper from Graph A (has the frozen entities' roles + states);
    # used for the picks and the recommendation quads too.
    chipper = _make_chipper(s4.get("graph_a", {}) or {})

    # (The recommendation + uncertainty labels are pinned ON the main scene
    # image — the big canvas — not here; see _stage4_pins in scene_component.)

    out.append("«sec:verdict»")
    # ── TRUST · the headline verdict: can we trust the VLM's advice so far? ──
    trust = s4.get("trust") or {}
    if trust and trust.get("score") is not None:
        tscore = trust.get("score")
        band = trust.get("band", "")
        bcol = ("#16a34a" if band == "high" else "#b45309" if band == "moderate"
                else "#b91c1c")
        trows: list[Any] = [
            html.Div([
                html.Span("TRUST · can we trust the VLM's advice so far?",
                          className="unc-tag"),
                html.Span(f"  {tscore} · {band}",
                          style={"fontSize": "13px", "fontWeight": "800",
                                 "color": bcol}),
            ], style={"marginBottom": "3px"}),
        ] + _trust_explanation_rows(trust, s4.get("alignment") or {}) \
          + _singular_error_rows(trust)
        # what this means → what to do (reliability of the advice, NOT the
        # emergency's urgency, and NOT groundedness)
        _WHATDO = {
            "high": ("rely on it as the model's stable position (not yet "
                     "proven grounded)", "#16a34a"),
            "moderate": ("usable — review the flagged recommendations before "
                         "acting", "#b45309"),
            "low": ("don't rely on it; the model contradicts itself — send to "
                    "reflection or a human", "#b91c1c")}
        wd, wdcol = _WHATDO.get(band, ("", "#64748b"))
        if wd:
            trows.append(html.Div(
                [html.Span("what to do: ", style={"fontWeight": "700",
                                                  "color": "#64748b"}),
                 html.Span(wd, style={"color": wdcol, "fontWeight": "600"})],
                style={"fontSize": "11.5px", "margin": "0 0 6px"}))
        # ranked contributors — WHY, worst-first. Only MATERIAL ones show by
        # default (≥ TRUST_SHOW_MIN); the rest collapse into a "minor" details,
        # so the card isn't a wall of near-zero factors.
        TRUST_SHOW_MIN = 0.05
        hits = [c for c in (trust.get("contributors") or [])
                if c.get("contribution", 0) > 0]
        major = [c for c in hits if c.get("contribution", 0) >= TRUST_SHOW_MIN]
        minor = [c for c in hits if c.get("contribution", 0) < TRUST_SHOW_MIN]

        # framing matches the verdict: under HIGH trust these are minor notes,
        # muted — not red "pull-downs" (Sunny: don't parade low-consequence
        # negatives beneath a green headline).
        high_band = band == "high"
        dent_col = "#94a3b8" if high_band else bcol

        def _contrib_row(c):
            return html.Div([
                html.Span(f"−{c.get('contribution')}",
                          style={"fontWeight": "700", "color": dent_col,
                                 "fontSize": "11.5px", "marginRight": "6px"}),
                html.Span(c.get("text", ""), style={"fontSize": "11.5px"}),
                html.Span(f"  ({c.get('evidence', '')})",
                          style={"fontSize": "10.5px", "color": "#94a3b8"}),
            ], style={"padding": "1px 0"})
        if major:
            trows.append(html.Div(
                "minor notes — didn't change the verdict:" if high_band
                else "what pulls trust down (worst first):",
                style={"fontSize": "11px", "color": "#64748b",
                       "fontWeight": "600"}))
            trows.extend(_contrib_row(c) for c in major)
        elif hits:
            trows.append(html.Div("no material concern — only minor factors",
                                  style={"fontSize": "11px", "color": "#16a34a"}))
        if minor:
            trows.append(html.Details([
                html.Summary(f"▸ {len(minor)} minor factor(s)",
                             style={"cursor": "pointer", "fontSize": "10.5px",
                                    "color": "#94a3b8", "listStyle": "none"}),
                html.Div([_contrib_row(c) for c in minor]),
            ], style={"marginTop": "2px"}))
        # what HOLDS trust up — the checks that passed clean (balance the card)
        _TRUST_POS = {"advice_backed_by_belief": "every danger it acts on is "
                                                 "one it holds",
                      "dangers_acted_on": "acts on every danger it sees",
                      "uncertainty": "stable on re-ask",
                      "internal_alignment": "hangs together",
                      "pick_agreement": "targets agree",
                      "conformance": "graph well-formed"}
        clean = [c for c in (trust.get("contributors") or [])
                 if c.get("contribution", 0) == 0]
        if clean:
            trows.append(html.Div(
                "holds up: " + " · ".join(_TRUST_POS.get(c["signal"], c["signal"])
                                          for c in clean),
                style={"fontSize": "11px", "color": "#16a34a",
                       "fontWeight": "600", "marginTop": "4px"}))
        # per-recommendation: ALL recs, best → worst trust. Two axes shown
        # side by side — trust (reliability) and consequence (life-safety).
        pr = [p for p in (trust.get("per_rec") or [])
              if p.get("score") is not None]
        if pr:
            trows.append(html.Div("every recommendation · trust (best → worst) "
                                  "· consequence to victims:",
                                  style={"fontSize": "11px", "color": "#64748b",
                                         "fontWeight": "600", "marginTop": "4px"}))

        def _tcol(sc):
            return ("#16a34a" if sc >= 0.7 else "#b45309" if sc >= 0.4
                    else "#b91c1c")

        def _ccol(bd):
            return {"high": "#b91c1c", "medium": "#b45309",
                    "low": "#16a34a"}.get(bd, "#94a3b8")
        for p in sorted(pr, key=lambda x: -x.get("score", 1.0)):
            w = p.get("worst_contributor") or {}
            cb = p.get("consequence_band")
            wv = p.get("worst_victim") or {}
            row = [
                html.Span(f"#{p.get('rank')} ", style={"fontWeight": "700",
                                                       "fontSize": "11.5px"}),
                chipper(p.get("threat", "")),
                html.Span(f" trust {p.get('score')}",
                          style={"fontSize": "11.5px",
                                 "color": _tcol(p.get("score", 1.0)),
                                 "fontWeight": "600", "margin": "0 4px"})]
            if cb:
                row.append(html.Span(
                    f"· consequence {cb}",
                    style={"fontSize": "11px", "color": _ccol(cb),
                           "fontWeight": "600", "marginRight": "4px"}))
                if wv.get("id"):
                    row.append(chipper(wv.get("id")))
            if w:
                # F37: name WHAT dented this recommendation, not the finding
                # itself. The finding renders in that card's own footer, and
                # printing it here too put the same sentence on the screen
                # twice — once above the cards and once under the card it is
                # about, which is the disconnect this reorganisation removes.
                _SIG = {"internal_alignment": "its parts don't line up",
                        "conformance": "a rule break",
                        "uncertainty": "unstable on re-ask",
                        "advice_backed_by_belief": "acts on a danger it "
                                                   "doesn't hold",
                        "dangers_acted_on": "sees a danger and skips it",
                        "pick_agreement": "targets disagree"}
                sig = str(w.get("signal", ""))
                row.append(html.Span(
                    f" — {_SIG.get(sig, sig)} (−{w.get('penalty')})",
                    style={"fontSize": "10.5px", "color": "#94a3b8"}))
            trows.append(html.Div(row, style={"padding": "1px 0"}))
        trows.append(html.Div(
            "trust = weighted average of the checks (the two A-vs-B "
            "directions weighted highest, 0.22 each). "
            "two axes: TRUST = can we rely on the advice (reliability); "
            "CONSEQUENCE = how bad for victims if the hazard isn't dealt with "
            "(life-safety). Neither is groundedness. Objective (no model, no "
            "answer key); weights are priors, to calibrate on the six scenes.",
            style={"fontSize": "10px", "color": "#cbd5e1", "marginTop": "5px",
                   "fontStyle": "italic"}))
        out.append(html.Div(trows, className="unc-panel",
                            style={"borderColor": bcol, "background": "#fff"}))

    picks = s4.get("picks", {}) or {}
    a = (picks.get("a_pick") or {}).get("threat") or "-"
    b = (picks.get("b_pick") or {}).get("threat") or "-"
    llm = (picks.get("llm_pick") or {}).get("threat") or "-"
    agree = picks.get("agreement")
    unan = picks.get("unanimous")
    col = "#16a34a" if unan else "#b45309"

    def _pk(label, val):
        el = chipper(val) if val and val != "-" else html.Span("-")
        return html.Div([html.Span(label, style={"color": "#64748b",
                                                  "fontSize": "11px",
                                                  "marginRight": "4px"}), el],
                        style={"padding": "2px 0"})
    out.append("«sec:picks»")
    out.append(html.Div([
        html.Span("SUPPRESSION TARGET (FOR THE CAUSAL TEST)",
                  className="unc-tag"),
        html.Div("Which mechanism is load-bearing — what we remove from the "
                 "scene to test whether the recommendation was grounded in "
                 "it. Not an instruction to a responder.",
                 style={"fontSize": "11px", "opacity": 0.75,
                        "padding": "2px 0 4px"}),
        _pk("algorithm · Graph A (out-degree):", a),
        _pk("model · Graph B (independent):", b),
        _pk("model · direct impact ask:", llm),
        html.Div(f"agreement {agree}  "
                 f"{'· all three agree' if unan else '· they disagree'}",
                 style={"color": col, "fontWeight": "700", "fontSize": "12px",
                        "marginTop": "4px"}),
    ], className="unc-panel", style={"borderColor": "#a78bfa",
                                     "background": "#faf5ff"}))

    out.append("«sec:stability»")
    # ── measured uncertainty (Phase 1b): how stable is the advice on re-ask? ──
    unc = s4.get("uncertainty") or {}
    if unc and unc.get("n_probes"):
        gran = unc.get("granular") or {}
        score = unc.get("score")
        scol = ("#16a34a" if (score or 0) < 0.2 else "#b45309"
                if (score or 0) < 0.5 else "#b91c1c")
        urows: list[Any] = [html.Div([
            html.Span("MEASURED UNCERTAINTY · do the recommendations hold up "
                      "on re-ask?", className="unc-tag"),
            # F46: was "score 0.446", which does not say which way is good.
            # This number RISES as the re-asks disagree with each other.
            html.Span(f"  {score:.2f} unsure · {unc.get('n_probes')} probes"
                      if isinstance(score, (int, float))
                      else f"  {score} · {unc.get('n_probes')} probes",
                      title=_UNSURE_TIP,
                      style={"fontSize": "11px", "color": scol,
                             "fontWeight": "700"}),
        ], style={"marginBottom": "3px"})]
        # the two scalar fields (top target, count). The top-target evidence
        # names entities, so render those as role-colored chips.
        for key, lbl in (("top_priority_target", "top-priority target"),
                         ("recommendation_count", "how many recommendations")):
            f = (gran.get("fields") or {}).get(key) or {}
            if f:
                u = f.get("u", 0)
                fc = "#16a34a" if u == 0 else "#b45309"
                ev = f.get("evidence", "")
                tail = (_votes_to_chips(ev, chipper)
                        if key == "top_priority_target"
                        else [html.Span(f"({ev})", style={"color": "#94a3b8",
                                                          "fontSize": "10.5px"})])
                urows.append(html.Div(
                    [html.Span(f"{lbl}: ", style={"color": "#64748b",
                                                  "fontSize": "11px"}),
                     html.Span(f"U {u}  ", style={"color": fc,
                                                  "fontWeight": "600",
                                                  "fontSize": "11px"})] + tail,
                    style={"padding": "1px 0"}))
        # per-entity threat wobble + per-threat effect (mechanism) wobble
        for tbl_key, tbl_lbl in (("threats", "entity flickers as a threat"),
                                 ("effects", "mechanism won't commit")):
            for oid, g in (gran.get(tbl_key) or {}).items():
                if not g.get("u"):
                    continue
                ev = g.get("evidence", g.get("votes", ""))
                urows.append(html.Div(
                    [chipper(oid),
                     html.Span(f" — {tbl_lbl}: U {g.get('u')} ({ev})",
                               style={"color": "#b45309", "fontSize": "11px"})],
                    style={"padding": "1px 0"}))
        # distinct recommendation sets across the re-asks
        cands = unc.get("candidates") or []
        if len(cands) > 1:
            urows.append(html.Div(
                f"the {unc.get('n_probes')} re-asks produced {len(cands)} "
                f"DIFFERENT recommendation sets "
                f"(votes: {', '.join(str(c.get('votes')) for c in cands)})",
                style={"fontSize": "11px", "color": "#b45309",
                       "fontWeight": "600", "marginTop": "3px"}))
        # (The dense driver narrative is intentionally NOT dumped here — the
        # structured rows above already say what's unstable and by how much.)
        _ro = (s4.get("runoff_judge") or {}).get("recommendations") or {}
        if _ro:
            _t = (_ro.get("text") or {}).get("verdict")
            urows.append(_bench_pointer(
                f"runoff: {_RO_LAB.get(_t, '?')}"
                + ("" if _ro.get("twins_agree") is None else
                   " · twins agree" if _ro.get("twins_agree")
                   else " · twins DISAGREE")))
        out.append(html.Div(urows, className="unc-panel",
                            style={"borderColor": "#c7d2fe",
                                   "background": "#eef2ff"}))

    # ── F37: the two score-only panels that used to sit here are gone.
    #
    # CONFORMANCE and INTERNAL ALIGNMENT (A) once held the findings. F24 moved
    # the card findings into card footers, F29 moved the set-level ones into
    # ACROSS ALL RECOMMENDATIONS, and F36 moved the graph ones onto the graphs
    # — leaving two panels printing a number above the things it described, and
    # INTERNAL ALIGNMENT still printing all five of its findings a second time.
    #
    # Each score now rides in the header of the section it scores, so no
    # finding and no number appears twice on one screen.

    out.append("«sec:recs»")
    # ── recommendations (the model's output, under test) ──
    #
    # F24: each card carries its OWN verdict in a footer. Before this, the
    # checks fired in one panel and the evidence sat in another — you could
    # read "5 internal-alignment failures" and have no way to see which card
    # they were about. A finding belongs under the thing it judges.
    explain = s4.get("explanation_alignment", {}) or {}
    by_rank = {k: list(v) for k, v in (explain.get("by_rank") or {}).items()}
    # The ALIGNMENT band must show EVERY alignment finding for this card, not
    # only F24's. internal_alignment's id-level failures carry a rank and were
    # rendering nowhere per-card: on D_aerial rec 1 the band read empty while
    # two real findings sat in the panel above, because F24's cross-checks
    # cannot fire when the reason does not parse.
    for f in ((s4.get("internal_alignment") or {}).get("failures") or []):
        if f.get("rank") is None:
            continue                      # set-level: at-risk coverage gaps
        by_rank.setdefault(str(f["rank"]), []).append(
            {**f, "signal": "internal_alignment", "level": "card",
             "rule": f.get("category", "")})
    mode_of = {str(m.get("rank")): m.get("mode", "")
               for m in (explain.get("modes") or [])}
    judged = {str(v.get("rank")): v
              for v in ((s4.get("card_judge") or {}).get("verdicts") or [])}
    for r in s4.get("recommendations", []):
        q = r.get("structured_reasoning", {}) or {}
        quad_row = ([html.Span("quad: ", style={"color": "#94a3b8",
                                                "fontSize": "11.5px"}),
                     chipper(q.get("threat", "")),
                     html.Span(f" —{q.get('effect', '')}→ ",
                               style={"color": "#7c3aed", "fontSize": "11.5px"})]
                    + [chipper(t) for t in (q.get("affected_objects") or [])])
        out.append(html.Div([
            html.Div([
                html.Span(f"#{r.get('rank')}", className="phase-num",
                          style={"marginRight": "6px"}),
                html.Span(r.get("action", ""), style={"fontWeight": "700"})],
                style={"marginBottom": "3px"}),
            html.Div(r.get("reason", ""), style={"fontSize": "12.5px",
                                                 "color": "#334155"}),
            html.Div(quad_row, style={"margin": "4px 0", "lineHeight": "1.8"}),
            html.Div(f"expected: {r.get('expected_consequence', '')}",
                     style={"fontSize": "11.5px", "color": "#64748b"}),
            html.Div(f"remaining risk: {r.get('remaining_risk', '')}",
                     style={"fontSize": "11.5px", "color": "#64748b"}),
            html.Div(f"follow-up: {r.get('possible_follow_up_action', '')}",
                     style={"fontSize": "11.5px", "color": "#94a3b8"}),
        ] + ([_rec_uncertainty_row(q.get("threat", ""), unc, chipper)]
             if _rec_uncertainty_row(q.get("threat", ""), unc, chipper) else [])
          + _card_verdict_rows(by_rank.get(str(r.get("rank")), []),
                               mode_of.get(str(r.get("rank")), ""),
                               None)
          + ([_bench_pointer(f"card judge ruled on rec {r.get('rank')}")]
             if judged.get(str(r.get("rank"))) else []),
           className="ticket", style={"margin": "6px 0"}))

    out.extend(_across_recommendations(s4))

    # ── the assumptions advisory (recorded, not in the graph) ──
    adv = s4.get("advisory", []) or []
    if adv:
        rows = [html.Div("ASSUMPTIONS ADVISORY · not in the graph",
                         className="unc-tag")]
        for a_ in adv:
            rows.append(html.Div(
                f"⚑ {a_.get('suspected', '')} "
                f"(anchor {a_.get('anchor_object_id', '')}) — "
                f"{a_.get('cue', '')} → {a_.get('suggested_action', '')}",
                style={"fontSize": "12px", "color": "#b45309",
                       "padding": "2px 0"}))
        out.append(html.Div(rows, className="unc-panel",
                            style={"borderColor": "#fde68a",
                                   "background": "#fffbeb"}))

    out.append("«sec:graphs»")
    # ── THE GRAPHS (F36) ────────────────────────────────────────────────
    #
    # Same shape as the recommendation cards: a finding renders WHERE the thing
    # it judges lives, and whatever compares two things sits BELOW both.
    #
    #   GRAPH A   the edges + graph_a's own conformance findings
    #   GRAPH B   the edges + graph_b's conformance, its self-consistency,
    #             whether its belief is stable over probes, and whether it is
    #             fit to be the yardstick at all
    #   A vs B    the comparison, below both
    #
    # Before this the graphs rendered as bare edge lists at the bottom of the
    # screen while five separate panels above judged them — the same
    # disconnect F24 fixed for the cards.
    #
    # ASYMMETRY, DELIBERATE: Graph B has a self-consistency band and Graph A
    # does not. Graph A is BUILT BY CODE from the quads, so it cannot
    # contradict itself the way a model's answer can; its only self-findings
    # are builder flags (unattached_hazard, unattributed_victim) and those
    # arrive through conformance. Sunny confirmed this is intentional.
    ga = s4.get("graph_a", {}) or {}
    gb = s4.get("graph_b", {}) or {}
    conf_issues = (s4.get("conformance") or {}).get("issues") or []

    def _graph_findings(which: str) -> list:
        mine = [i for i in conf_issues if i.get("graph") == which]
        if not mine:
            return [html.Div("✓ no conformance issues",
                             style={"fontSize": "11.5px", "color": "#16a34a",
                                    "padding": "1px 0"})]
        # Grouped by ENTITY, because two rules firing on one edge is ONE
        # defect counted twice. Printing them as separate lines makes a single
        # mistake look like two — the annotation says so explicitly.
        by_ent: dict[str, list] = {}
        for i in mine:
            by_ent.setdefault(i.get("entity") or "—", []).append(i)
        rows = []
        for ent, its in sorted(by_ent.items(),
                               key=lambda kv: -max(i.get("severity", 0)
                                                   for i in kv[1])):
            rows.append(html.Div(
                chipper(ent) if ent != "—" else html.Span(
                    "graph-level", style={"fontSize": "11px",
                                          "color": "#64748b"}),
                style={"marginTop": "3px"}))
            for i in sorted(its, key=lambda x: -x.get("severity", 0)):
                sev = i.get("severity", 0)
                rows.append(html.Div(
                    ("○ " if sev == 0 else "⚠ ")
                    + f"{i.get('rule', '')}: {i.get('detail', '')}",
                    style={"fontSize": "11.5px", "padding": "1px 0 1px 14px",
                           "color": "#94a3b8" if sev == 0 else
                                    ("#b91c1c" if sev >= 2 else "#b45309")}))
            if len(its) > 1:
                rows.append(html.Div(
                    f"the same edge, {len(its)} rules — one defect",
                    style={"fontSize": "10px", "color": "#94a3b8",
                           "fontStyle": "italic", "paddingLeft": "14px"}))
        return rows

    def _band(title: str) -> Any:
        return html.Div(title, style={"fontSize": "11px", "color": "#64748b",
                                      "fontWeight": "600", "marginTop": "6px"})

    by_g = (s4.get("conformance") or {}).get("by_graph") or {}
    gate = ((s4.get("trust") or {}).get("graph_b_gate") or {})

    def _score_line(which: str) -> Any:
        gg = by_g.get(which) or {}
        n = gg.get("count", 0)
        return html.Span(
            f"  {n} issue(s)" + (f" · worst severity {gg.get('max_severity', 0)}"
                                 if n else ""),
            style={"fontSize": "11px", "color": "#64748b"})

    warns = ga.get("graph_warnings", []) or []
    out.append(causal_graph_view(ga, "GRAPH A · from the recommendations"))
    a_rows = [html.Div([_band("CONFORMANCE — is this graph well-formed?"),
                        _score_line("graph_a")])]
    a_rows += _graph_findings("graph_a")
    if warns:
        a_rows.append(html.Div(f"⚠ {len(warns)} builder warning(s): {warns[0]}",
                               style={"fontSize": "11.5px", "color": "#b45309"}))
    a_rows.append(html.Div(
        "built by code from the recommendation quads — it cannot contradict "
        "itself, so it has no self-consistency band",
        style={"fontSize": "10px", "color": "#cbd5e1",
               "fontStyle": "italic", "marginTop": "4px"}))
    out.append(html.Div(a_rows, className="unc-panel"))

    out.append(causal_graph_view(gb, "GRAPH B · the model's own belief"))
    b_rows = [html.Div([_band("CONFORMANCE — is this graph well-formed?"),
                        _score_line("graph_b")])]
    b_rows += _graph_findings("graph_b")
    # The GATE is a verdict about Graph B's fitness, so it belongs under Graph
    # B — not buried in the trust panel, where a failed gate showed up only as
    # "signals_measured 4/5" with no way to see why.
    # F39: shown only when it FAILS. A passing gate is one green line saying
    # nothing happened, on every clean run; the value is entirely in the
    # failure case, where it explains why A-vs-B vanished from trust.
    if not gate.get("trusted", True):
        b_rows.append(_band("FIT TO BE THE YARDSTICK?"))
        for why in gate.get("reasons", []):
            b_rows.append(html.Div(
                f"⛔ no — {why}",
                style={"fontSize": "11.5px", "color": "#b91c1c"}))
        b_rows.append(html.Div(
            "A-vs-B is withheld from trust: comparing against an unsound "
            "yardstick gives a meaningless number, not a low one",
            style={"fontSize": "10px", "color": "#cbd5e1",
                   "fontStyle": "italic"}))
    out.append(html.Div(b_rows, className="unc-panel"))

    # ── Graph B self-consistency (does B contradict its own declarations) ──
    gbi = s4.get("graph_b_internal") or {}
    if gbi and gbi.get("measured"):
        rows_i = [html.Div([
            html.Span("INTERNAL ALIGNMENT (B) · does Graph B agree with "
                      "itself?", className="unc-tag"),
            # F46: "score 0.778" -> the same word the cards use, so a reader
            # can tell it is the same kind of number pointing the same way.
            html.Span(f"  {gbi.get('n_failures')} issue(s) · "
                      f"{float(gbi.get('score', 0)):.2f} clean",
                      title=_CLEAN_TIP,
                      style={"fontSize": "11px", "color": "#64748b"}),
        ], style={"marginBottom": "3px"})]
        for r in (gbi.get("breakdown") or []):
            sev = r.get("severity", 1)
            col = ("#b91c1c" if sev >= 2 else "#b45309" if sev == 1
                   else "#94a3b8")
            rows_i.append(html.Div(
                [html.Span(r.get("category", ""),
                           style={"fontWeight": "700", "color": col,
                                  "fontSize": "12px"}),
                 html.Span(f"  ×{r.get('count')}",
                           style={"color": "#64748b", "fontSize": "11px"})],
                style={"margin": "2px 0"}))
            for ex in (r.get("examples") or []):
                rows_i.append(html.Div(ex, style={"fontSize": "10.5px",
                                                  "color": "#94a3b8",
                                                  "paddingLeft": "10px"}))
        if not (gbi.get("breakdown") or []):
            rows_i.append(html.Div("clean — Graph B is self-consistent",
                                   style={"fontSize": "12px",
                                          "color": "#16a34a"}))
        out.append(html.Div(rows_i, className="unc-panel"))

    out.append("«sec:stability»")
    # ── Graph B UNCERTAINTY (F42) ──────────────────────────────────────
    #
    # Does the model reproduce its own causal graph when asked again?
    #
    # This panel used to print EVERY edge any probe produced — sixteen lines on
    # D_aerial, eleven of them seen once, plus a mechanism line per pair, plus
    # the picks. A wall. What a reader needs is three things: what the model
    # BELIEVES (the edges a majority of probes repeat), where it CONTRADICTS
    # itself, and whether it invented entities. The once-only tail is noise by
    # definition — that is what the instability score already summarises.
    gbu = s4.get("graph_b_uncertainty") or {}
    if gbu.get("n_probes"):
        n = gbu.get("n_probes")
        rows_b = [html.Div([
            html.Span("GRAPH B UNCERTAINTY · does the model reproduce its own "
                      "causal graph?", className="unc-tag"),
            # F46: the score itself was missing — the panel showed the probe
            # count and three per-axis agreements, but not the number the
            # yardstick gate and the pathology layer actually read.
            html.Span(f"  {gbu['score']:.2f} unsure · {n} probes"
                      if isinstance(gbu.get("score"), (int, float))
                      else f"  {n} probes",
                      title=_UNSURE_TIP,
                      style={"fontSize": "11px", "color": "#64748b"}),
        ], style={"marginBottom": "3px"})]

        def _line(txt, col="#334155", pad=0, size="11.5px"):
            rows_b.append(html.Div(txt, style={
                "fontSize": size, "color": col,
                "padding": f"1px 0 1px {pad}px"}))

        # the three numbers, in words. An instability of 0.2 means the model
        # agreed with itself 80% of the time — printing the raw figure next to
        # a label like "direction" left the reader to work out which way is
        # good.
        for key, good, bad in (
                ("direction_instability",
                 "always points the arrows the same way",
                 "flips which end is the hazard"),
                ("edge_set_instability",
                 "draws the same links each time",
                 "draws different links each time"),
                ("pick_instability",
                 "picks the same thing to suppress",
                 "picks a different thing to suppress")):
            v = gbu.get(key)
            if not isinstance(v, (int, float)):
                continue
            agree = round((1 - v) * 100)
            _line(f"{agree}% agreement — " + (good if v <= 0.2 else bad),
                  "#16a34a" if v <= 0.2 else
                  ("#b45309" if v <= 0.5 else "#b91c1c"))

        ev = [e for e in (gbu.get("direction_evidence") or [])
              if isinstance(e, dict)]
        eff = gbu.get("effect_evidence") or {}
        held = [e for e in ev if (e.get("votes") or 0) * 2 > (e.get("of") or n)]
        if held:
            _line("WHAT IT CONSISTENTLY BELIEVES", "#64748b", size="11px")
            for e in sorted(held, key=lambda x: -(x.get("votes") or 0)):
                src, tgt = str(e.get("source", "")), str(e.get("target", ""))
                verbs = ", ".join(sorted((eff.get(f"{src}->{tgt}") or {}))) or "—"
                rows_b.append(html.Div(
                    [chipper(src), html.Span(" → "), chipper(tgt),
                     html.Span(f"  ({verbs})  {e.get('votes')}/{e.get('of', n)}"
                               f" probes",
                               style={"color": "#94a3b8",
                                      "fontSize": "10.5px"})],
                    style={"fontSize": "11.5px", "padding": "1px 0 1px 10px"}))
        else:
            _line("no link survives a majority of re-asks — the model does not "
                  "hold a stable causal picture of this scene", "#b91c1c")

        pk = gbu.get("pick_evidence") or {}
        if pk:
            top = max(pk.items(), key=lambda kv: kv[1])
            _line(f"would suppress {top[0]} in {top[1]} of {n} probes"
                  + (f" · also considered {', '.join(k for k in pk if k != top[0])}"
                     if len(pk) > 1 else ""), "#4f46e5")

        once = len(ev) - len(held)
        if once:
            _line(f"{once} further link(s) appeared in a minority of probes "
                  f"and are not shown", "#cbd5e1", pad=10, size="10px")

        for c in (gbu.get("both_directions_in_one_probe") or []):
            _line(f"⚠ one probe asserts BOTH directions of "
                  f"{c.get('a')} ↔ {c.get('b')} in the same answer — the model "
                  f"contradicts itself inside one response", "#b91c1c")
        for f in (gbu.get("flags") or []):
            _line("⚠ " + str(f.get("evidence", "")), "#b91c1c")

        inv = gbu.get("invented_ids") or []
        if inv:
            _line(f"⚠ the model named {len(inv)} entit(y/ies) the scene does "
                  f"not have: {', '.join(sorted(inv))}", "#b91c1c")

        _line("re-asked at raised temperature. This is whether the model's own "
              "BELIEF is stable — the yardstick A is measured against.",
              "#cbd5e1", size="10px")
        _rog = (s4.get("runoff_judge") or {}).get("graph_b") or {}
        if _rog:
            _t = (_rog.get("text") or {}).get("verdict")
            rows_b.append(_bench_pointer(
                f"runoff: {_RO_LAB.get(_t, '?')}"
                + ("" if _rog.get("twins_agree") is None else
                   " · twins agree" if _rog.get("twins_agree")
                   else " · twins DISAGREE")))
        out.append(html.Div(rows_b, className="unc-panel"))

    # ── ALIGNMENT: does the advice match the model's own beliefs? ──
    # Graph A = the causal links the advice LEANS ON. Graph B = the links the
    # model INDEPENDENTLY declares. This panel asks whether they agree — a
    # self-consistency (faithfulness) check: does the model agree with itself.
    # It is NOT on Pearl's rung 2. Rung 2 is DOING — an intervention: remove
    # the hazard from the scene, re-run, check the advice changes (CEE+'s core
    # groundedness test, a later stage). We change NOTHING in the scene here;
    # we only compare two things the model already said. This is a prerequisite
    # BELOW the ladder — and, because it compares the model against itself, it
    # needs no ground truth and can run at runtime.
    out.append("«sec:alignment»")
    al = s4.get("alignment") or {}
    _gate = ((s4.get("trust") or {}).get("graph_b_gate") or {})
    # F44: the numbers are ALWAYS shown. "NOT COMPUTED" was untrue — the
    # comparison is computed and stored in every run; only TRUST withholds it.
    # Hiding the panel meant the worse graph escaped measurement: on D_aerial,
    # Graph B named the two hazmat workers as the victims and failed the gate
    # on reproducibility, while Graph A protected three vehicles and ignored
    # the people — and, because the yardstick was disqualified, nothing
    # measured Graph A at all. The warning rides at the top instead.
    if al:
        _warn = ([html.Div(
            "⚠ Graph B failed the yardstick check, so these numbers are "
            "shown but WITHHELD from trust — read them as a lead, not a "
            "measurement.",
            style={"fontSize": "11px", "color": "#b91c1c",
                   "fontWeight": "600", "margin": "2px 0"})]
            + [html.Div(f"· {r}", style={"fontSize": "10.5px",
                                         "color": "#b91c1c",
                                         "paddingLeft": "8px"})
               for r in _gate.get("reasons", [])]) \
            if _gate and not _gate.get("trusted", True) else []
        a_only = al.get("a_only") or []      # in the advice, model doesn't back
        b_only = al.get("b_only") or []      # model believes, no advice acts on
        af, bc = al.get("a_fidelity"), al.get("b_coverage")
        if af is not None and bc is not None:
            if af >= 0.8 and bc >= 0.8:
                verdict = ("the advice and the model's own causal beliefs tell "
                           "the SAME story — self-consistent")
                vcol = "#16a34a"
            else:
                verdict = ("the advice and the model's own causal beliefs "
                           "DIVERGE — it recommends links it doesn't "
                           "independently hold, and/or holds dangers it "
                           "never acts on")
                vcol = "#b45309"
        else:
            verdict, vcol = "", "#64748b"
        rows2: list[Any] = [
            html.Div([
                html.Span("ALIGNMENT · does the advice match the "
                          "model's own causal beliefs?", className="unc-tag"),
            ]),
            html.Div(verdict, style={"fontSize": "12px", "color": vcol,
                                     "fontWeight": "600", "margin": "2px 0 5px"}),
        ] + _warn + _resolved_id_rows(al.get("resolved_ids") or {},
                                           al.get("resolved_by") or {}) + [
            # F43: NAME them. The panel showed only the plain-English gloss,
            # so nothing on screen connected to "a_fidelity < 0.4 fires
            # sycophancy" or to the `ab_alignment` row in trust — Sunny went
            # looking for both numbers and could not find them. The gloss
            # stays; the name is what lets a reader follow it anywhere else.
            html.Div([html.B(f"a_fidelity {af} "),
                      "— of the causal claims the advice leans on, how many "
                      "the model's own graph backs ",
                      html.Span("(low = asserted only to justify actions)",
                                style={"color": "#94a3b8"})],
                     style={"fontSize": "11.5px"}),
            html.Div([html.B(f"b_coverage {bc} "),
                      "— of the claims the model believes, how many the advice "
                      "acts on ",
                      html.Span("(low = dangers it sees but didn't act on)",
                                style={"color": "#94a3b8"})],
                     style={"fontSize": "11.5px"}),
        ] + _ab_decomposition_rows(al.get("decomposition") or {}) + \
            _effect_toggle_rows(al) + \
            ([_bench_pointer("graph judge spoke on this comparison")]
             if s4.get("graph_judge") else []) + [
            # F43: `overall agreement` removed — a third number derived from
            # a_fidelity and b_coverage, printed directly beneath both.
        ]
        # the ACTUAL disagreeing edges — the whole point of the reframe
        # F40: collapsed to ENTITY PAIRS, worst hazard first. Four "asserted
        # not believed" lines and three "believed not acted on" lines described
        # about three disagreements, repeated with different effect words. The
        # consequential unit is who-endangers-whom; the verb goes in brackets.
        # F43: the edge-level dumps are gone. Fourteen lines — three of them
        # the same claim about one hazard and three vehicles — restating what
        # the two entity lines above already say, with the verb and a repeated
        # severity added. The entity lines name WHO; the verbs are covered by
        # the mechanism splits on the card and by Graph B uncertainty.
        if not a_only and not b_only:
            rows2.append(html.Div("every link lines up — nothing to show",
                                  style={"fontSize": "11px", "color": "#16a34a",
                                         "marginTop": "4px"}))
        # F43: the Pearl's-rung-2 footnote printed on every run and belongs in
        # the docs, not on the screen.
        out.append(html.Div(rows2, className="unc-panel"))

    # F37: the overall conformance number and Arm A's frozen raw numbers lost
    # their panel when the findings moved. They are a ROLLUP across the graphs
    # and the cards, so they sit at the foot of the graph section rather than
    # above the things they summarise. The Arm A raw pair is kept because the
    # three-arm comparison depends on it.
    out.append("«sec:graphs»")
    _conf = s4.get("conformance") or {}
    if _conf:
        out.append(html.Div(
            # F46: one score covers BOTH graphs and the cards together — there
            # is no per-graph conformance number, which is why the CONFORMANCE
            # band under each graph shows counts and not a score. Saying so
            # here stops a reader hunting for the missing one.
            f"conformance, both graphs and the cards together (there is no "
            f"per-graph number): {_conf.get('n_issues')} issue(s) · "
            f"{float(_conf.get('validity', 0)):.2f} clean  ·  Arm A raw "
            f"(frozen, saturates): A={_conf.get('raw_a_validity')} "
            f"B={_conf.get('raw_b_validity')} — kept for comparison",
            title=_CLEAN_TIP,
            style={"fontSize": "10px", "color": "#94a3b8", "marginTop": "4px"}))
    return _assemble_stage4_sections(out, s4, d)


def assess_component(d: dict[str, Any], unc_view: str = "both",
                     record_name: str | None = None,
                     judge_result: dict[str, Any] | None = None) -> list[Any]:
    """PHASE 1 · ASSESS body: verdict banner, the two uncertainty channels
    (view chosen by the toggle: model's self-report vs measured vs both,
    side-by-side being the calibration view), causal drivers, and the
    stage's activity timeline — same grammar as the Phase 0 stations."""
    A = d["assess"]
    if A["status"] == "pending" and not A.get("petition"):
        return [html.Div("waiting for perception to finish...",
                         className="ticket-empty")]
    out: list[Any] = []

    # RAG SHADOW (rag/both retrieval mode only): exact-key vs RAG top-1 for
    # the rules Stage 2 (assess + reflection) actually quoted. The auto
    # result diff rides along here — it is a Stage 2 concept (it re-runs the
    # verdict). Empty when mode is 'exact' or no rules were quoted.
    panel = rag_shadow_panel(d.get("retrieval_shadow"), "Stage 2 · assess",
                             result_diff=d.get("retrieval_result_diff"),
                             backend_reason=d.get("retrieval_backend_reason"))
    if panel is not None:
        out.append(panel)

    # Medium-bound derivations: code overrode a state the model gave,
    # with the victim that caused it. Shown first — it changes the
    # evidence the whole assessment reads.
    for line in A.get("derived_hazards") or []:
        out.append(html.Div(
            f"⚑ DERIVED HAZARD — {line}",
            style={"background": "#fef2f2", "border": "1px solid #ef4444",
                   "borderRadius": "8px", "padding": "5px 10px",
                   "margin": "2px 0 6px", "fontSize": "12px",
                   "color": "#b91c1c", "fontWeight": "600"}))

    v = A.get("verdict")
    if v:
        color = BUCKET_CSS.get(v.get("bucket"), "#64748b")
        text = (f"{'DISASTER' if v.get('scenario') == 'Yes' else 'NO DISASTER'}"
                f" · {v.get('disaster_type')} · level {v.get('level')} · "
                f"{str(v.get('bucket', '')).upper()}")
        out.append(html.Div(text, className="verdict-banner",
                            style={"background": f"{color}18",
                                   "border": f"1px solid {color}",
                                   "color": color, "fontWeight": "700",
                                   "borderRadius": "10px",
                                   "padding": "8px 14px",
                                   "margin": "4px 0 8px"}))

    # Threats and at-risk lists (merged stage), each entity wearing its
    # granular membership-U so the SOURCE of instability is pinpointable.
    mu = A.get("uncertainty")
    gran = (mu or {}).get("granular") or {}

    def entity_rows(entries, table, color, kind_label):
        erows = []
        for e in entries or []:
            oid = e.get("object_id", "?")
            g = (table or {}).get(oid, {})
            u = g.get("u")
            if u is None and mu is not None and mu.get("n_probes"):
                # In the final answer but in ZERO probe lists: maximal
                # membership instability, not "no data" (house_1 case).
                u = 1.0
                g = {"votes": f"0/{mu.get('n_probes')}"}
            badge = None
            if u is not None:
                badge = html.Span(
                    f"U {u} ({g.get('votes', '')})",
                    style={"marginLeft": "auto", "fontSize": "11px",
                           "fontWeight": "700",
                           "color": "#b45309" if u > 0.2 else "#16a34a"})
            kind = e.get("kind")
            desc = (f" · {kind}" if kind else "") + \
                   (f" — {e.get('reason')}" if e.get("reason") else "")
            erows.append(html.Div(
                [html.Span("●", style={"color": color, "marginRight": "6px"}),
                 html.Span(oid, style={"fontWeight": "700"}),
                 html.Span(desc, style={"color": "#64748b",
                                        "fontSize": "12px",
                                        "marginLeft": "4px"})]
                + ([badge] if badge is not None else []),
                style={"display": "flex", "alignItems": "baseline",
                       "padding": "2px 0"}))
        if erows:
            erows.insert(0, html.Div(kind_label, className="unc-tag",
                                     style={"marginTop": "6px"}))
        return erows

    if v:
        out += entity_rows(v.get("threats"), gran.get("threats"),
                           "#ef4444", "THREATS")
        out += entity_rows(v.get("at_risk"), gran.get("at_risk"),
                           "#f97316", "AT RISK")
        # Entities that FLICKER in probes but missed the canonical lists
        # are instability the canonical answer hides — surface them.
        cited = {e.get("object_id") for e in
                 (v.get("threats") or []) + (v.get("at_risk") or [])}
        ghosts = [(oid, g) for table in
                  (gran.get("threats") or {}, gran.get("at_risk") or {})
                  for oid, g in table.items() if oid not in cited]
        for oid, g in ghosts:
            out.append(html.Div(
                f"◌ {oid} appeared in {g.get('votes')} probe lists but NOT "
                f"in the final answer — unstable membership",
                style={"fontSize": "12px", "color": "#b45309",
                       "padding": "2px 0 2px 4px"}))

    # The uncertainty panel: channel 1 (model's claim) vs channel 2 (ours).
    rows: list[Any] = []
    if unc_view in ("self", "both") and v is not None:
        claim = v.get("self_confidence")
        rows.append(html.Div([
            html.Span("model says", className="unc-tag"),
            html.Span("—" if claim is None else f"{claim:.2f} confident",
                      className="unc-val"),
            html.Span("(self-reported: recorded, never trusted)",
                      className="unc-note"),
        ], className="unc-row"))
    if unc_view in ("measured", "both") and mu is not None:
        rows.append(html.Div([
            html.Span("we measured", className="unc-tag"),
            html.Span(f"U = {mu.get('score')}", className="unc-val"),
            html.Span(f"disaster scenario {mu.get('scenario_agreement')} · "
                      f"type {mu.get('type_agreement')} · "
                      f"bucket {mu.get('bucket_agreement')} "
                      f"({mu.get('n_probes') or len(A.get('probes', []))} probes)",
                      className="unc-note"),
        ], className="unc-row"))
        if mu.get("explanation"):
            rows.append(html.Div(f"why: {mu['explanation']}",
                                 className="unc-why"))
    if unc_view == "both" and mu is not None and v is not None \
            and v.get("self_confidence") is not None:
        gap = round(abs(v["self_confidence"] - (1 - mu.get("score", 0))), 2)
        rows.append(html.Div(
            f"calibration gap: claims {v['self_confidence']:.2f}, probes "
            f"support {1 - mu.get('score', 0):.2f} (gap {gap})",
            className="unc-gap",
            style={"color": "#b45309" if gap > 0.15 else "#16a34a"}))
    if rows:
        out.append(html.Div(rows, className="unc-panel"))

    # Assessment tickets: same lifecycle grammar as Stage 1 (OPEN ->
    # FIXING during a reflection round -> FIXED / STOOD). Body quotes the
    # rulebook chunk, so the ticket shows WHICH law and WHY.
    from agentic.rulebook import retrieve as _retrieve
    for tk in A.get("tickets", {}).values():
        status = tk["status"]
        stamp = {"open": ("OPEN", "stamp open"),
                 "fixing": ("FIXING…", "stamp fixing"),
                 "fixed": ("FIXED", "stamp fixed"),
                 "stood": ("STOOD ITS GROUND", "stamp stood")}[status]
        chunk = _retrieve(tk["kind"])
        body_lines = [html.Div("EVIDENCE", className="speaker rulebook"),
                      html.Div(tk["evidence"], className="bubble rulebook-bubble")]
        if chunk:
            body_lines += [
                html.Div(f"RULE {chunk.rule_id}", className="speaker rulebook"),
                html.Div(f"{chunk.rule} — {chunk.rationale}",
                         className="bubble rulebook-bubble")]
        out.append(html.Details([
            html.Summary([
                html.Span(str(tk["kind"]).replace("_", " "),
                          className="ticket-kind"),
                html.Span(stamp[0], className=stamp[1]),
            ]),
            html.Div(body_lines, className="ticket-body"),
        ], className=f"ticket {status}"))

    # The reflection ledger: rounds, verdicts, and the U0->U1 strip.
    R = A.get("reflection")
    if R and (R.get("rounds") or R.get("u_after") is not None):
        rrows: list[Any] = [html.Div("REFLECTION", className="unc-tag",
                                     style={"marginTop": "8px"})]
        for r in R.get("rounds", []):
            glyph = ("✓" if r.get("changed")
                     else "■" if r.get("changed") is False else "…")
            verdict = ("revised" if r.get("changed")
                       else "stood its ground" if r.get("changed") is False
                       else "in flight")
            # THE JUDGES of this round, as cards: every mechanism that
            # contributed a verdict, and what it decided (Sunny: not just
            # printed text — separate cards per judge).
            trigs = r.get("triggers", [])
            viols = [t for t in trigs if t.get("type") == "violation"]
            membs = [t for t in trigs if t.get("type") == "membership_split"]
            fields = [t for t in trigs if t.get("type") == "field_instability"]

            def _card(title, color, lines):
                return html.Div(
                    [html.Div(title, style={
                        "fontSize": "9px", "fontWeight": "800",
                        "letterSpacing": ".08em", "color": "#fff",
                        "background": color, "padding": "2px 8px",
                        "borderRadius": "6px 6px 0 0"})]
                    + [html.Div(x, style={"fontSize": "10px",
                                          "padding": "1px 8px",
                                          "color": "#334155"})
                       for x in lines],
                    style={"border": f"1px solid {color}55",
                           "borderRadius": "6px", "background": "#fff",
                           "minWidth": "120px", "maxWidth": "180px",
                           "paddingBottom": "4px"})

            from agentic.rulebook import retrieve as _rb
            cards: list[Any] = []
            if viols:
                cards.append(_card("⚖ CODE CHECKS", "#dc2626",
                                   [f"found: {vt.get('kind', '?').replace('_', ' ')}"
                                    for vt in viols]))
            rule_ids: list[str] = []
            for vt in viols:
                ch = _rb(vt.get("kind", ""))
                if ch and ch.rule_id not in rule_ids:
                    rule_ids.append(ch.rule_id)
            if membs:
                rule_ids += [x for x in ("G1", "G3") if x not in rule_ids]
            if fields:
                rule_ids += [x for x in ("S1",) if x not in rule_ids]
            if rule_ids:
                cards.append(_card("⚖ RULEBOOK", "#7c3aed",
                                   [f"cited: {', '.join(rule_ids)}"]))
            if membs:
                cards.append(_card("⚖ PROBE METER", "#0ea5e9",
                                   [f"{m.get('object_id')}: only "
                                    f"{m.get('votes')} lists"
                                    for m in membs]))
                cards.append(_card("⚖ GEOMETRY", "#16a34a",
                                   ["nominated candidates from",
                                    "declared boxes (2D hints)"]))
            if fields:
                cards.append(_card("⚖ PROBE METER" if not membs else
                                   "⚖ FIELD STABILITY", "#f59e0b",
                                   [f"{f.get('driver', '?')}: "
                                    f"{str(f.get('evidence', ''))[:48]}"
                                    for f in fields]))
            detail: list[Any] = []
            if cards:
                detail.append(html.Div(cards, style={
                    "display": "flex", "gap": "6px", "flexWrap": "wrap",
                    "margin": "4px 0"}))
            if r.get("instruction"):
                detail.append(html.Div("WHAT THE RULEBOOK SENT:",
                                       className="unc-tag",
                                       style={"marginTop": "4px"}))
                detail.append(html.Pre(
                    r["instruction"],
                    style={"fontSize": "10px", "whiteSpace": "pre-wrap",
                           "maxHeight": "220px", "overflowY": "auto",
                           "background": "#f1f5f9", "padding": "8px",
                           "borderRadius": "8px", "margin": "2px 0"}))
            rrows.append(html.Details([
                html.Summary(f"{glyph} round {r.get('round')}: "
                             f"{r.get('summary')} — {verdict}",
                             style={"fontSize": "12px", "cursor": "pointer"}),
                html.Div(detail, style={"padding": "2px 0 4px 12px"}),
            ], style={"padding": "1px 0"}))
        # Verdict diff: what reflection actually changed, [0] -> [-1].
        hist = A.get("verdict_history", [])
        if len(hist) > 1:
            def _ids(v, key):
                return {t.get("object_id", "?") for t in v.get(key) or []}

            def vline(v):
                th = ",".join(sorted(_ids(v, "threats"))) or "-"
                ar = ",".join(sorted(_ids(v, "at_risk"))) or "-"
                return (f"{v.get('scenario')} · {v.get('disaster_type')} · "
                        f"L{v.get('level')} | threats: {th} | at-risk: {ar}")

            # WHAT CHANGED, as chips: +added (green) / -removed (red).
            chips: list[Any] = [html.Span("VERDICT CHANGE:",
                                          className="unc-tag")]
            b, a_ = hist[0], hist[-1]
            if (b.get("scenario"), b.get("level")) != (a_.get("scenario"),
                                                       a_.get("level")):
                chips.append(html.Span(
                    f"{b.get('scenario')}·L{b.get('level')} → "
                    f"{a_.get('scenario')}·L{a_.get('level')}",
                    style={"fontWeight": "700", "fontSize": "11px",
                           "marginRight": "6px"}))
            for key in ("threats", "at_risk"):
                added = _ids(a_, key) - _ids(b, key)
                removed = _ids(b, key) - _ids(a_, key)
                for oid in sorted(added):
                    chips.append(html.Span(
                        f"+{oid} ({key})",
                        style={"background": "#dcfce7", "color": "#166534",
                               "borderRadius": "6px", "padding": "1px 6px",
                               "fontSize": "11px", "fontWeight": "700",
                               "marginRight": "4px"}))
                for oid in sorted(removed):
                    chips.append(html.Span(
                        f"−{oid} ({key})",
                        style={"background": "#fee2e2", "color": "#991b1b",
                               "borderRadius": "6px", "padding": "1px 6px",
                               "fontSize": "11px", "fontWeight": "700",
                               "marginRight": "4px"}))
            if len(chips) == 1:
                chips.append(html.Span("wording/reasons only",
                                       style={"fontSize": "11px",
                                              "color": "#64748b"}))
            rrows.append(html.Div([
                html.Div(chips, style={"marginTop": "4px"}),
                html.Div("BEFORE: " + vline(hist[0]),
                         style={"fontSize": "11px", "color": "#94a3b8"}),
                html.Div("AFTER:    " + vline(hist[-1]),
                         style={"fontSize": "11px", "fontWeight": "600",
                                "color": "#0f172a"}),
            ], style={"marginTop": "4px", "borderTop": "1px dashed #e2e8f0",
                      "paddingTop": "4px"}))
        stamp = {"clean": ("CLEAN", "#16a34a"),
                 "no_change": ("STOOD", "#64748b"),
                 "cap_reached": ("CAP REACHED", "#b45309"),
                 "model_error": ("MODEL ERROR", "#dc2626")}.get(
            R.get("stopped"), ("", "#64748b"))
        line: list[Any] = [html.Span(stamp[0], style={
            "fontWeight": "700", "color": stamp[1], "fontSize": "12px"})]
        if R.get("u_after") is not None:
            delta = round((R.get("u_before") or 0) - R["u_after"], 3)
            arrow_color = "#16a34a" if delta > 0 else "#b45309"
            line.append(html.Span(
                f"  U {R.get('u_before')} → {R['u_after']} "
                f"({'−' if delta > 0 else '+'}{abs(delta)})",
                style={"fontWeight": "700", "color": arrow_color,
                       "marginLeft": "10px", "fontSize": "12px"}))
            line.append(html.Span(
                " (ΔU alone is not proof of improvement — check the "
                "verdict change)",
                style={"color": "#94a3b8", "fontSize": "11px",
                       "marginLeft": "6px"}))
        rrows.append(html.Div(line, style={"marginTop": "2px"}))
        out.append(html.Div(rrows, className="unc-panel"))

    # ── The petition panel: Stage 2's ledgered look-back at Stage 1 ────
    P = A.get("petition")
    if P:
        pcolor = {"in_flight": "#7c3aed", "merged": "#16a34a",
                  "failed": "#dc2626"}.get(P.get("status"), "#64748b")
        ptitle = ("PETITION → SAME STAGE (question re-asked fresh, cap 1 "
                  "— the image was fine, the sorting was questioned)"
                  if P.get("target") == "stage2" else
                  "PETITION → STAGE 1 (contextual re-perception, cap 1)")
        prows: list[Any] = [html.Div(
            ptitle, className="unc-tag",
            style={"marginTop": "8px", "color": pcolor})]
        for r in P.get("reasons", []):
            prows.append(html.Div(
                f"reason: {str(r.get('kind', '')).replace('_', ' ')} — "
                f"{str(r.get('evidence', ''))[:90]}",
                style={"fontSize": "12px"}))
        if P.get("status") == "in_flight":
            prows.append(html.Div("re-perceiving the image…",
                                  className="working",
                                  style={"fontSize": "12px"}))
        elif P.get("status") == "merged":
            for x in P.get("added", []):
                prows.append(html.Span(f"+{x}", style={
                    "background": "#ede9fe", "color": "#5b21b6",
                    "borderRadius": "6px", "padding": "1px 6px",
                    "fontSize": "11px", "fontWeight": "700",
                    "marginRight": "4px"}))
            for x in P.get("removed", []):
                prows.append(html.Span(f"−{x}", style={
                    "background": "#fee2e2", "color": "#991b1b",
                    "borderRadius": "6px", "padding": "1px 6px",
                    "fontSize": "11px", "fontWeight": "700",
                    "marginRight": "4px"}))
            for x in P.get("disputed", []):
                prows.append(html.Div(
                    f"⚠ second look omitted {x} — DISPUTED, not erased "
                    f"(petitions add, never delete)",
                    style={"fontSize": "11px", "color": "#b45309"}))
            if P.get("note"):
                prows.append(html.Div(P["note"],
                                      style={"fontSize": "11px",
                                             "color": "#64748b"}))
        elif P.get("status") == "failed":
            prows.append(html.Div(
                f"FAILED: {P.get('error', '')[:80]} — recorded as "
                f"pathology signal, run proceeds on the original record",
                style={"fontSize": "12px", "color": "#dc2626"}))
        oc = P.get("outcome")
        if oc:
            prows.append(html.Div(
                ("✓ RESOLVED — the violation pressure vanished on the "
                 "merged record" if oc.get("resolved") else
                 f"■ UNRESOLVED — still standing: {oc.get('after')}"),
                style={"fontSize": "12px", "fontWeight": "700",
                       "color": "#16a34a" if oc.get("resolved")
                       else "#b45309", "marginTop": "3px"}))
        elif P.get("status") in ("merged", "failed") and not P.get("added"):
            # The second look added nothing, so stage 2 never re-ran and no
            # outcome event is coming. Silence here read as "still working";
            # say plainly that the stage is finished, that the first-pass
            # verdict stands, and that whatever triggered the petition is
            # still open — nothing has been re-tested against it.
            _open = ", ".join(
                str(r.get("kind", "")).replace("_", " ")
                for r in P.get("reasons", [])) or "the original pressure"
            _ev = "; ".join(str(r.get("evidence", ""))[:90]
                            for r in P.get("reasons", []) if r.get("evidence"))
            prows.append(html.Div(
                "■ NO CHANGE — the second look added nothing, so Stage 2 was "
                "not re-run and the first-pass verdict STANDS.",
                style={"fontSize": "12px", "fontWeight": "700",
                       "color": "#b45309", "marginTop": "3px"}))
            prows.append(html.Div(
                f"still unresolved: {_open}"
                + (f" — {_ev}" if _ev else ""),
                style={"fontSize": "11px", "color": "#b45309"}))
        # Cross-epoch comparison: the petition's evidentiary payoff.
        e0 = d.get("epoch0")
        after_v = A.get("verdict")          # never trust loop-scope names

        def _short(vv):
            th = ",".join(t.get("object_id", "?")
                          for t in vv.get("threats") or []) or "-"
            return (f"{vv.get('scenario')}·L{vv.get('level')} "
                    f"threats[{th}]")

        hist0 = ((e0 or {}).get("assess") or {}).get("verdict_history") or []

        def _full(vv):
            th = ", ".join(t.get("object_id", "?")
                           for t in vv.get("threats") or []) or "none"
            ar = ", ".join(
                f"{r.get('object_id', '?')}"
                + (f" ({r.get('kind')})" if r.get("kind") else "")
                for r in vv.get("at_risk") or []) or "none"
            return (f"{vv.get('scenario')} · level {vv.get('level')} · "
                    f"threats: {th} · at-risk: {ar}")

        # THE ENDING, always (Sunny: no scrolling back to find what the
        # decision ended up being). One grey BEFORE line, one loud
        # FINAL line — whether the answer changed or stood.
        final_v = after_v or (hist0[-1] if hist0 else None)
        if hist0 and final_v:
            prows.append(html.Div([
                html.Span("BEFORE: ", className="unc-tag"),
                html.Span(_full(hist0[-1]),
                          style={"fontSize": "11px", "color": "#94a3b8"}),
            ], style={"marginTop": "5px"}))
            changed = (after_v is not None
                       and _full(after_v) != _full(hist0[-1]))
            fcolor = BUCKET_CSS.get((final_v or {}).get("bucket"),
                                    "#334155")
            prows.append(html.Div(
                ("FINAL DECISION: " if changed
                 else "FINAL DECISION (unchanged): ") + _full(final_v),
                style={"background": f"{fcolor}14",
                       "border": f"1px solid {fcolor}",
                       "color": fcolor, "fontWeight": "700",
                       "fontSize": "12px", "borderRadius": "8px",
                       "padding": "6px 10px", "marginTop": "4px"}))
        out.append(html.Div(prows, className="unc-panel",
                            style={"borderColor": pcolor}))

    # ── THE JUDGES: task-allocation roster, every mechanism that judges
    # this stage, as cards (Sunny). Uniform anatomy: TASK / AUTHORITY /
    # THIS RUN. Iron rule on every card: judges flag, measure, and
    # advise — none may overwrite the model's answer; only reflection
    # carries the message and only the model revises.
    def _jcard(title, color, task, authority, verdict_lines, dev=False):
        head_style = {"fontSize": "10px", "fontWeight": "800",
                      "letterSpacing": ".08em", "color": "#fff",
                      "background": color, "padding": "3px 10px",
                      "borderRadius": "8px 8px 0 0"}
        rows = [html.Div(("⚗ " if dev else "⚖ ") + title, style=head_style),
                html.Div([html.Span("TASK ", className="unc-tag"),
                          html.Span(task, style={"fontSize": "10px"})],
                         style={"padding": "3px 8px 0"}),
                html.Div([html.Span("AUTHORITY ", className="unc-tag"),
                          html.Span(authority,
                                    style={"fontSize": "10px",
                                           "fontStyle": "italic"})],
                         style={"padding": "0 8px"})]
        rows.append(html.Div([html.Span("THIS RUN ", className="unc-tag")]
                             + [html.Div(x, style={"fontSize": "11px",
                                                   "fontWeight": "600",
                                                   "color": "#0f172a"})
                                for x in verdict_lines],
                             style={"padding": "0 8px 6px"}))
        return html.Div(rows, style={
            "border": f"1px solid {color}66", "borderRadius": "8px",
            "background": "#fff", "width": "215px",
            "boxShadow": "0 1px 4px rgba(15,23,42,.06)"})

    cards: list[Any] = []
    tickets = list(A.get("tickets", {}).values())
    if A["status"] != "pending":
        by_status: dict[str, int] = {}
        for tk in tickets:
            by_status[tk["status"]] = by_status.get(tk["status"], 0) + 1
        tk_lines = ([f"{tk['kind'].replace('_', ' ')} → {tk['status'].upper()}"
                     for tk in tickets[:3]]
                    + ([f"+{len(tickets) - 3} more"] if len(tickets) > 3
                       else [])) or ["clean — nothing to flag"]
        cards.append(_jcard(
            "CODE CHECKS", "#dc2626",
            "decidable rule violations (S1-S7, G2, G4)",
            "flags -> triggers reflection; never edits",
            tk_lines))
        cited = []
        for tk in tickets:
            from agentic.rulebook import retrieve as _rb2
            ch = _rb2(tk["kind"])
            if ch and ch.rule_id not in cited:
                cited.append(ch.rule_id)
        R0 = A.get("reflection") or {}
        for r in R0.get("rounds", []):
            for tg in r.get("triggers", []):
                if tg.get("type") == "membership_split":
                    cited += [x for x in ("G1", "G3") if x not in cited]
                if tg.get("type") == "field_instability" and "S1" not in cited:
                    cited.append("S1")
        rb_lines = []
        for tk in tickets[:2]:
            from agentic.rulebook import retrieve as _rb3
            ch3 = _rb3(tk["kind"])
            if ch3:
                rb_lines.append(f"{ch3.rule_id} → for "
                                f"{tk['kind'].replace('_', ' ')}")
        extra_rules = [x for x in cited if x not in
                       {ln.split(" ")[0] for ln in rb_lines}]
        if extra_rules:
            rb_lines.append(f"{', '.join(extra_rules)} → for probe splits")
        cards.append(_jcard(
            "RULEBOOK", "#7c3aed",
            "states each rule: what it is, why, and an example",
            "advises — words only, quoted into reflection",
            rb_lines or ["not consulted"]))
        if mu is not None:
            n_split = sum(1 for dr in (mu.get("drivers") or [])
                          if "split" in str(dr))
            gran0 = mu.get("granular") or {}
            splits = []
            for lname in ("threats", "at_risk"):
                for oid, g in (gran0.get(lname) or {}).items():
                    if g.get("u", 0) > 0.2:
                        splits.append(f"{oid}: {g.get('votes')} "
                                      f"{lname} lists")
            cards.append(_jcard(
                "PROBE METER", "#0ea5e9",
                "stability: re-asks the same question, counts votes",
                "measures -> triggers reflection above U 0.2",
                [f"U = {mu.get('score')} over {mu.get('n_probes')} probes"]
                + splits[:3]
                + ([f"+{len(splits) - 3} more splits"]
                   if len(splits) > 3 else [])))
        ctx0 = A.get("context") or {}
        nh = ctx0.get("n_spatial_hints")
        pair_lines = list(ctx0.get("spatial_pairs") or [])[:3]
        if nh and len(ctx0.get("spatial_pairs") or []) > 3:
            pair_lines.append(f"+{nh - 3} more pairs")
        cards.append(_jcard(
            "GEOMETRY", "#16a34a",
            "nominates proximity candidates from declared boxes",
            "advises — nominates, never convicts (G3)",
            pair_lines or (["no pairs near a hazard"] if nh == 0
                           else ["hints fed into the prompt"])))
        # Talking points: computed live, judge-free.
        v_now = A.get("verdict")
        if v_now and d.get("result"):
            try:
                from agentic.evals import citation_counts
                cc = citation_counts(_verdict_to_assessment(v_now),
                                     d["result"])
                rate = cc.get("state_citation_rate")
                tp_lines = [("no cited entities" if rate is None else
                             f"{cc['state_cited']}/{cc['n_entries']} reasons "
                             f"cite their state ({rate:.0%})")]
                if cc.get("uncited"):
                    tp_lines.append("vague: "
                                    + ", ".join(cc["uncited"][:4]))
                if cc.get("victim_shaped"):
                    tp_lines.append(
                        f"{len(cc['victim_shaped'])} threat reason(s) "
                        f"victim-shaped (S8): "
                        + ", ".join(cc["victim_shaped"][:3]))
                cards.append(_jcard(
                    "TALKING POINTS", "#f59e0b",
                    "do the reasons cite the declared states?",
                    "measures a groundedness proxy; informs only",
                    tp_lines))
            except Exception:
                pass
        ro = A.get("runoff")
        if ro:
            win = ro.get("winner")
            w_line = ro.get(f"{win}_line", "?") if win else "?"
            l_key = "top2" if win == "top1" else "top1"
            l_line = ro.get(f"{l_key}_line", "?")
            cards.append(_jcard(
                "RUNOFF (LLM)", "#be185d",
                "high U: blind-judges the model's own top-2 readings",
                "advises — winner becomes reflection context",
                [f"judged: {w_line} ({ro.get(win + '_votes', '?')})",
                 f"    vs: {l_line} ({ro.get(l_key + '_votes', '?')})",
                 f"→ preferred the first: "
                 f"{str(ro.get('raw', ''))[:60]}"]))
        # Pairwise LLM judge: on demand.
        if judge_result is None:
            jlines: list[Any] = ["not yet run"]
        elif judge_result.get("status") == "running":
            jlines = ["deliberating…"]
        else:
            jlines = []
            if judge_result.get("pre_line"):
                jlines += [f"judged: pre  {judge_result['pre_line']}",
                           f"    vs: post {judge_result.get('post_line', '?')}"]
            jlines += [f"winner: {judge_result.get('winner', '?').upper()} — "
                       f"{str(judge_result.get('raw', ''))[:60]}"]
        cards.append(_jcard(
            "PAIRWISE (LLM)", "#334155",
            "pre vs post, blind A/B: which fits the declared states?",
            "advises; different training family than the subject",
            jlines))
        # GT: the only judge that knows the answer — dev-only.
        ge = gt_eval_overlay(d, record_name)
        if ge is not None:
            q = ge["quadrant"]
            gl = [f"error {ge['pre']['error_score']} -> "
                  f"{ge['post']['error_score']}  ·  {q}"]
            s1g = ge.get("stage1") or {}
            for m in (s1g.get("missed_entities") or [])[:2]:
                gl.append(f"S1 MISS: {m.get('label')} — petition target")
            if ge.get("matched_by") == "image_name":
                gl.append("(matched by image name)")
            cards.append(_jcard(
                "GROUND TRUTH", "#0f172a",
                "the verified answer — calibration scenes only",
                "measures; NEVER exists in production",
                gl, dev=True))
        else:
            cards.append(_jcard(
                "GROUND TRUTH", "#94a3b8",
                "the verified answer — calibration scenes only",
                "measures; NEVER exists in production",
                ["no GT for this scene (production shape)"], dev=True))
    if cards:
        judge_children: list[Any] = [html.Div(cards, style={
            "display": "flex", "gap": "8px", "flexWrap": "wrap",
            "margin": "6px 0"})]
        if judge_result is None and gt_eval_overlay(d, record_name):
            judge_children.append(html.Button(
                "⚖ RUN PAIRWISE JUDGE (pre vs post, blind)",
                id={"type": "judge-btn", "n": 0}, n_clicks=0,
                className="go", style={"fontSize": "11px"}))
        out.append(html.Details([
            html.Summary("THE JUDGES — task allocation",
                         className="unc-tag",
                         style={"cursor": "pointer", "marginTop": "8px"}),
            html.Div(judge_children),
        ], open=True))

    acts = A.get("activities", [])
    if acts:
        color = "#0ea5e9"
        rows2: list[Any] = []
        shown = acts[-7:]
        if len(acts) > 7:
            rows2.append(html.Div([
                html.Div("⋯", className="tl-node dim",
                         style={"borderColor": "#cbd5e1"}),
                html.Div(f"{len(acts) - 7} earlier steps", className="tl-text dim"),
            ], className="tl-row"))
        for i, a in enumerate(shown):
            now = (A["status"] == "active" and i == len(shown) - 1)
            glyph, mod = _timeline_glyph(a)
            style = {"borderColor": color}
            if now:
                glyph, mod = "▸", "now"
                style = {"borderColor": color, "color": color,
                         "boxShadow": f"0 0 0 4px {color}22"}
            rows2.append(html.Div([
                html.Div(glyph, className=f"tl-node {mod}", style=style),
                html.Div(a, className=f"tl-text {mod}"),
            ], className="tl-row"))
        out.append(html.Div(rows2, className="tl",
                            style={"--tl-line": f"{color}55"}))
    return out


def tickets_component(d: dict[str, Any]) -> html.Div:
    """Violation tickets with lifecycle stamps; expanded = the Rulebook's
    exact instruction (the Model side lives in the round summary)."""
    cards = []
    for key, v in d["violations"].items():
        status = v["status"]
        # An open violation during an in-flight repair round is literally
        # being fixed right now; its ticket says so and pulses.
        if status == "open" and d.get("round_in_progress"):
            status = "fixing"
        stamp = {"open": ("OPEN", "stamp open"),
                 "fixing": ("FIXING…", "stamp fixing"),
                 "fixed": ("FIXED", "stamp fixed"),
                 "stood": ("STOOD ITS GROUND", "stamp stood")}[status]
        cards.append(html.Details([
            html.Summary([
                html.Span(v["kind"].replace("_", " "), className="ticket-kind"),
                html.Span(f"'{v['raw_label']}'", className="ticket-label"),
                html.Span(stamp[0], className=stamp[1]),
            ]),
            html.Div([
                html.Div("RULEBOOK", className="speaker rulebook"),
                html.Div(v["instruction"], className="bubble rulebook-bubble"),
            ], className="ticket-body"),
        ], className=f"ticket {status}"))
    if not cards:
        # Three empty states, not two. `petition_started` clears violations
        # and stop_reason so the re-perception can build its own panels, so
        # after a petition the first branch fails and the old fallback claimed
        # the run had not started yet — on a second pass that had already
        # finished. An empty ticket list means something different depending
        # on where in the run we are, and the panel has to say which.
        # Any petition status at all means we are past the first pass — do not
        # enumerate them ('merged', 'failed', 'in_flight', and whatever a later
        # route adds), or a new status silently restores the wrong message.
        _pet = ((d.get("assess") or {}).get("petition") or {}).get("status")
        if d["stop_reason"] == "clean" and not d["violations"]:
            label = "no violations — clean on arrival"
        elif _pet:
            label = ("re-perceiving — waiting for the second answer..."
                     if _pet == "in_flight"
                     else "re-perception raised no violations")
        else:
            label = "waiting for the model's first answer..."
        cards = [html.Div(label, className="ticket-empty")]
    # RAG shadow for STAGE 1 (repair): exact-key vs RAG top-1 for the P1-P6
    # rules repair quoted. Same panel grammar as Stage 2, minus the answer
    # diff (repair has no verdict to move). Empty unless rag/both mode.
    panel = rag_shadow_panel(d.get("retrieval_shadow_perceive"),
                             "Stage 1 · repair",
                             backend_reason=d.get("retrieval_backend_reason"))
    if panel is not None:
        cards = [panel] + cards
    return html.Div(cards, className="tickets")


def pipeline_steps(d: dict[str, Any]) -> tuple[list[tuple[str, str]],
                                               list[tuple[str, str]]]:
    """Every pipeline step with its status ('pending'/'active'/'done'),
    split by stage: (stage-1 steps, stage-2 steps)."""
    A = d["assess"]
    s1: list[tuple[str, str]] = [(s, d["stages"][s]["status"])
                                 for s in STAGES]
    # A stage-1 petition blanks the live assessment so the re-run can build
    # its own panels — but when the second look adds nothing, stage 2 never
    # re-runs and the blank dict is all that is left. Answer/Probes/Reflect
    # DID run; the record is in epoch0. Reading the blank made a finished
    # stage report 'pending' and the badge pulse forever.
    _pet = A.get("petition") or {}
    if (A.get("status", "pending") == "pending" and _pet
            and _pet.get("status") != "in_flight"):
        _prior = (d.get("epoch0") or {}).get("assess") or {}
        if _prior.get("status") == "done" or _prior.get("verdict"):
            A = _prior
    a_status = A.get("status", "pending")
    if a_status == "pending":
        answer = probes = reflect = "pending"
    else:
        answer = "done" if A.get("verdict") else "active"
        probes = ("done" if A.get("uncertainty") is not None
                  else "active" if A.get("probes") else "pending")
        R = A.get("reflection") or {}
        reflect = ("done" if R.get("stopped")
                   else "active" if R.get("rounds") else "pending")
        if a_status == "done":
            answer = probes = reflect = "done"
    s2 = [("Answer", answer), ("Probes", probes), ("Reflect", reflect)]
    pet = A.get("petition") or _pet
    if pet:
        s2.append(("Second look",
                   "active" if pet.get("status") == "in_flight"
                   else "done"))
    return s1, s2


def phase_status_span(steps: list[tuple[str, str]],
                      now_text: str = "",
                      settled: bool = False) -> html.Span:
    """The little live badge on a stage card's header:
    '✓ done' · '● <step it is on>…' (pulsing) · '○ waiting'.

    `settled` says the stage has finished even though not every step ran.
    Without it there was no state meaning "done, but partial", so a finished
    stage fell into the partial branch, got className 'active', and pulsed
    forever — the CSS reads 'active' as still working."""
    if any(s == "failed" for _, s in steps):
        # terminal and not working: never pulse. The CSS reads 'active' as
        # still in progress, which is what made a dead run look alive.
        failed = [n for n, s in steps if s == "failed"][0]
        return html.Span(f"✕ failed at {failed.lower()}",
                         className="phase-status")
    if steps and all(s == "done" for _, s in steps):
        return html.Span("✓ done", className="phase-status done")
    active = [n for n, s in steps if s == "active"]
    if active and not settled:
        label = now_text or active[0].lower()
        done_n = sum(1 for _, s in steps if s == "done")
        return html.Span(f"● {label} · step {done_n + 1}/{len(steps)}",
                         className="phase-status active")
    if any(s == "done" for _, s in steps):
        done_n = sum(1 for _, s in steps if s == "done")
        if settled:
            # terminal, not in progress: no pulse.
            return html.Span(f"✓ done · {done_n}/{len(steps)} steps",
                             className="phase-status done partial")
        return html.Span(f"● {done_n}/{len(steps)} steps",
                         className="phase-status active")
    return html.Span("○ waiting", className="phase-status pending")


def progress_strip(d: dict[str, Any]) -> html.Div:
    """All pipeline steps as chips (kept for tests/reuse; Sunny prefers
    the per-card badges, so this no longer renders at the top)."""
    s1, s2 = pipeline_steps(d)
    steps = s1 + s2
    chips: list[Any] = []
    for i, (name, status) in enumerate(steps):
        if i:
            done_link = steps[i - 1][1] == "done"
            chips.append(html.Div(
                className="pstrip-link" + (" done" if done_link else "")))
        # F31: a failed stage gets its own glyph. Without it a died-mid-run
        # stage fell through to "○ waiting", which is the opposite of true.
        glyph = {"done": "✓", "active": "●", "failed": "✕"}.get(status, "○")
        chips.append(html.Div(
            [html.Div(glyph, className=f"pchip-dot {status}"),
             html.Div(name, className=f"pchip-name {status}")],
            className=f"pchip {status}"))
    rows: list[Any] = [html.Div(chips, className="pstrip-row")]
    # One plain line under the strip: what is happening right now.
    act = d.get("activity") or {}
    active = [n for n, s in steps if s == "active"]
    failed = [n for n, s in steps if s == "failed"]
    if failed:
        rows.append(html.Div(f"failed at {failed[0].lower()} — {d.get('error', '')}",
                             style={"fontSize": "11.5px", "color": "#be123c",
                                    "marginTop": "4px"}))
    elif active and act.get("text"):
        rows.append(html.Div(f"now: {act['text']}",
                             className="pstrip-now"))
    elif all(s == "done" for _, s in steps) and steps:
        rows.append(html.Div("all stages complete",
                             className="pstrip-now done"))
    return html.Div(rows, className="pstrip")


def instruments_component(d: dict[str, Any]) -> html.Div:
    n_entities = (len(d["result"]["detected_objects"]) if d["result"]
                  else len(d["anchors"]) or len(d["repair_entities_latest"]))
    open_v = sum(1 for v in d["violations"].values() if v["status"] == "open")
    fixed_v = sum(1 for v in d["violations"].values() if v["status"] == "fixed")
    stood_v = sum(1 for v in d["violations"].values() if v["status"] == "stood")
    matched = sum(1 for b in d["bound"].values() if b["box_source"] == "dino_matched")
    fallback = sum(1 for b in d["bound"].values() if b["box_source"] == "vlm_sam_fallback")

    def tile(value: Any, label: str, cls: str = "") -> html.Div:
        return html.Div([html.Div(str(value), className="tile-value"),
                         html.Div(label, className="tile-label")],
                        className=f"tile {cls}")

    tiles = [
        tile(n_entities, "entities"),
        tile(open_v, "open", "amber" if open_v else ""),
        tile(fixed_v, "fixed", "green" if fixed_v else ""),
        tile(stood_v, "stood", "gray" if stood_v else ""),
        tile("●" * d["rounds_used"] + "○" * max(0, 2 - d["rounds_used"]), "rounds"),
        tile(f"{matched}/{matched + fallback}" if (matched + fallback) else "–", "bound"),
    ]
    # Stage 2 tiles appear once a verdict exists: bucket + measured U.
    v = d["assess"].get("verdict")
    mu = d["assess"].get("uncertainty")
    if v:
        tiles.append(tile(str(v.get("bucket", "?")).upper(), "bucket"))
    if mu is not None:
        u = mu.get("score", 0)
        tiles.append(tile(f"{u}", "measured U",
                          "amber" if u and u > 0.2 else "green"))
    return html.Div(tiles, className="instruments")


def _valid_box(bbox: Any) -> bool:
    """True only for a well-formed [x1, y1, x2, y2] with positive area.

    Mid-repair entities are the model's RAW answer; a malformed bbox (one
    element, strings, inverted corners) must be skipped by the renderer,
    never crash it (live-run ValueError, 2026-07-21)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _pct_box(bbox: list[int], size: list[int]) -> dict[str, str]:
    """Percent-position a VALID box, clamped into the image frame (raw
    model boxes may exceed the image; clamping keeps overlays inside)."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    left = min(max(100 * x1 / w, 0.0), 100.0)
    top = min(max(100 * y1 / h, 0.0), 100.0)
    right = min(max(100 * x2 / w, 0.0), 100.0)
    bottom = min(max(100 * y2 / h, 0.0), 100.0)
    return {"left": f"{left:.2f}%", "top": f"{top:.2f}%",
            "width": f"{max(right - left, 0.0):.2f}%",
            "height": f"{max(bottom - top, 0.0):.2f}%"}


def scene_component(d: dict[str, Any], image_src: str | None) -> list[Any]:
    """The image with live overlays: dashed anchors, snapped final boxes,
    amber chips for open violations. Boxes are clickable (inspector).

    Returns the CHILDREN for the layout's .scene container. The container
    itself (id="scene", className="scene") lives in the layout and carries
    position:relative + overflow:hidden; the percentage-positioned boxes
    are meaningless outside it (v0 bug: rendering these children into a
    classless div anchored the boxes to the page)."""
    if not image_src:
        return [html.Div("upload an image or pick a replay", className="scene-empty")]
    size = d["image_size"]
    children: list[Any] = [html.Img(src=image_src, className="scene-img")]
    # The image narrates the run: activity ribbon + scanning sweep while a
    # model is thinking, spotlight on the entity being worked on.
    act = d.get("activity") or {}
    if act.get("busy"):
        children.append(html.Div(className="sweep"))
    if act.get("text"):
        children.append(html.Div(act["text"],
                                 className="ribbon busy" if act.get("busy") else "ribbon"))
    # Verdict ribbon (Stage 2): the scene wears its assessment, tinted by
    # bucket, docked at the bottom edge; pulses while assess is active.
    A = d.get("assess") or {}
    v = A.get("verdict")
    if v:
        color = BUCKET_CSS.get(v.get("bucket"), "#64748b")
        label = ("NO DISASTER" if v.get("scenario") != "Yes"
                 else f"DISASTER · {v.get('disaster_type')}")
        busy = A.get("status") == "active"
        children.append(html.Div(
            f"{label} · level {v.get('level')} · "
            f"{str(v.get('bucket', '')).upper()}",
            className="verdict-ribbon" + (" busy" if busy else ""),
            style={"position": "absolute", "left": "12px", "bottom": "12px",
                   "background": f"{color}e6", "color": "#fff",
                   "fontWeight": "700", "fontSize": "13px",
                   "letterSpacing": "0.04em", "padding": "6px 12px",
                   "borderRadius": "8px",
                   "boxShadow": f"0 2px 12px {color}66"}))
    elif A.get("status") == "active":
        children.append(html.Div(
            "assessing scene from declared state...",
            className="verdict-ribbon busy",
            style={"position": "absolute", "left": "12px", "bottom": "12px",
                   "background": "#0ea5e9e6", "color": "#fff",
                   "fontWeight": "700", "fontSize": "13px",
                   "padding": "6px 12px", "borderRadius": "8px"}))
    final_by_id = {o["object_id"]: o for o in
                   (d["result"] or {}).get("detected_objects", [])}
    if size:
        # Paint order: largest boxes first so smaller ones sit ON TOP and
        # stay visible/clickable (Sunny: road_1 was burying spill_1).
        def _area(a: dict[str, Any]) -> float:
            oid = a.get("object_id")
            fin = final_by_id.get(oid) or {}
            bnd = d["bound"].get(oid) or {}
            box = (fin.get("bbox") or bnd.get("bbox") or a.get("anchor_bbox"))
            if not _valid_box(box):
                return 0.0
            return float((box[2] - box[0]) * (box[3] - box[1]))

        for a in sorted(d["anchors"], key=_area, reverse=True):
            oid = a.get("object_id")
            bound = d["bound"].get(oid)
            final = final_by_id.get(oid)
            spot = " spotlight" if act.get("oid") == oid else ""
            if final and not _valid_box(final.get("bbox")):
                final = None
            if bound and not _valid_box(bound.get("bbox")):
                bound = None
            if final and final.get("bbox"):
                color = STATE_KIND_CSS.get(final["state_kind"], "#94a3b8")
                dashed = final["box_source"] == "vlm_sam_fallback"
                style = _pct_box(final["bbox"], size)
                style["borderColor"] = color
                if dashed:
                    style["borderStyle"] = "dashed"
                children.append(html.Div(
                    html.Span(f"{oid} · {final['state']}",
                              className="box-tag", style={"background": color}),
                    id={"type": "scene-box", "oid": oid},
                    className=f"scene-box {final['state_kind']}{spot}",
                    style=style, n_clicks=0))
            elif bound and bound.get("bbox"):
                style = _pct_box(bound["bbox"], size)
                children.append(html.Div(
                    html.Span(oid, className="box-tag"),
                    id={"type": "scene-box", "oid": oid},
                    className=f"scene-box bound{spot}", style=style, n_clicks=0))
            elif _valid_box(a.get("anchor_bbox")):
                style = _pct_box(a["anchor_bbox"], size)
                children.append(html.Div(className="scene-box anchor", style=style))
        # Amber chips for open violations that point at a listed entity.
        ents = d["repair_entities_latest"] or d["anchors"]
        for v in d["violations"].values():
            if v["status"] != "open":
                continue
            idx = v.get("entity_index", -1)
            chip_txt = f"{v['kind'].replace('_', ' ')}: '{v['raw_label']}'"
            if 0 <= idx < len(ents):
                box = ents[idx].get("bbox") or ents[idx].get("anchor_bbox")
                if _valid_box(box):
                    style = _pct_box(box, size)
                    children.append(html.Div(chip_txt, className="chip",
                                             style={"left": style["left"], "top": style["top"]}))
                    continue
            children.append(html.Div(chip_txt, className="chip docked"))
        # Stage 4: pin the recommendation + uncertainty labels ONTO this same
        # canvas (not a separate image), once the run has a Stage-4 result.
        s4 = d.get("stage4")
        if s4:
            children.extend(_stage4_pins(d, _make_chipper(s4.get("graph_a", {})
                                                          or {})))
    return children


def inspector_component(d: dict[str, Any], selected: str | None):
    """Entity inspector as a modal popup (the space under the image is
    reserved for the conversation agent). Returns None when nothing is
    selected, which renders an empty modal container."""
    result = d["result"]
    if not selected or not result:
        return None
    obj = next((o for o in result["detected_objects"] if o["object_id"] == selected), None)
    if not obj:
        return None
    color = STATE_KIND_CSS.get(obj["state_kind"], "#94a3b8")
    chain = []
    if obj.get("label_note"):
        chain.append(f"model said something else first: {obj['label_note']}")
    if obj.get("family_name_as_label"):
        chain.append("flagged: family name used as label")
    if obj.get("vocab_extension"):
        chain.append("outside the vocabulary (escape hatch)")
    chain.append(f"geometry: {obj['box_source']}"
                 + (f" (conf {obj['box_confidence']:.2f})" if obj.get("box_confidence") else ""))
    if obj.get("anchor_bbox") and obj.get("bbox") and obj["anchor_bbox"] != obj["bbox"]:
        chain.append(f"anchor {obj['anchor_bbox']} -> final {obj['bbox']}")
    return html.Div(
        html.Div([
            html.Div([html.Span(obj["object_id"], className="insp-id"),
                      html.Span(obj["state"], className="insp-state",
                                style={"background": color}),
                      # Pattern-matching id: a plain id would break the click
                      # callback whenever the modal is closed (Dash never
                      # fires a callback whose Input component is absent;
                      # ALL-pattern inputs tolerate zero matches).
                      html.Button("×", id={"type": "insp-close", "n": 0},
                                  className="insp-close", n_clicks=0)],
                     className="insp-head"),
            html.Div(obj.get("description", ""), className="insp-desc"),
            html.Ul([html.Li(c) for c in chain], className="insp-chain"),
        ], className="modal"),
        className="modal-backdrop")


# ── App assembly ────────────────────────────────────────────────────────

# suppress_callback_exceptions: the inspector modal's close button only
# exists while the modal is open; Dash must tolerate its absence otherwise.
app = dash.Dash(__name__, title="CEE+ Agentic — Pipeline",
                suppress_callback_exceptions=True)

app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  /* Light theme: airy background, white cards, soft layered shadows. The
     signal colors (red hazard / orange at-risk / blue normal / amber
     repair / green clean) stay identical to the overlay language. */
  :root { color-scheme: light;
    --bg:#eef2f7; --card:#ffffff; --line:#e2e8f0; --ink:#1e293b;
    --muted:#64748b; --faint:#94a3b8; --accent:#2563eb;
    --shadow:0 1px 2px rgba(15,23,42,.06), 0 8px 24px rgba(15,23,42,.08);
    --shadow-lift:0 2px 6px rgba(15,23,42,.08), 0 16px 40px rgba(15,23,42,.14); }
  body { background:var(--bg); color:var(--ink);
      font-family:'Helvetica Neue',Arial,sans-serif; margin:0; }
  .wrap { max-width:1500px; margin:0 auto; padding:16px 22px; }
  h1 { font-size:17px; letter-spacing:2px; color:var(--accent); }
  .controls { display:flex; gap:10px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
  .controls .upload { border:1px dashed #cbd5e1; background:var(--card); border-radius:10px;
      padding:9px 16px; cursor:pointer; color:var(--muted); box-shadow:var(--shadow); }
  .upload-inner { display:flex; align-items:center; gap:9px; }
  .upload-thumb { height:34px; border-radius:6px; display:block; box-shadow:0 1px 3px rgba(0,0,0,.25); }
  .upload-name { font-size:12px; color:var(--ink); max-width:180px; overflow:hidden;
      text-overflow:ellipsis; white-space:nowrap; }
  .controls input[type=text] { background:var(--card); border:1px solid var(--line); color:var(--ink);
      border-radius:10px; padding:9px 12px; width:420px; box-shadow:var(--shadow); }
  .go { background:var(--accent); color:white; border:none; border-radius:10px; padding:10px 20px;
      font-weight:bold; letter-spacing:1px; cursor:pointer; box-shadow:var(--shadow); }
  .go:hover { filter:brightness(1.08); }
  .instruments { display:flex; gap:10px; margin:10px 0; }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:12px;
      padding:8px 16px; min-width:72px; text-align:center; box-shadow:var(--shadow); }
  .tile-value { font-size:19px; font-weight:bold; color:var(--ink); }
  .tile-label { font-size:10px; color:var(--faint); letter-spacing:1px; text-transform:uppercase; }
  .tile.amber .tile-value { color:#d97706; } .tile.green .tile-value { color:#16a34a; }
  .tile.gray .tile-value { color:var(--faint); }
  .main { display:grid; grid-template-columns: 1fr 540px; gap:20px; }
  .scene { position:relative; border-radius:14px; overflow:visible; background:#0f172a;
      box-shadow:var(--shadow-lift); }
  .scene-img { width:100%; display:block; border-radius:14px; }
  .scene-empty, .ticket-empty { color:var(--faint); padding:34px; text-align:center;
      background:var(--card); border-radius:12px; border:1px dashed var(--line); }
  .scene-box { position:absolute; border:3px solid #94a3b8; border-radius:4px; cursor:pointer; }
  .scene-box.anchor { border:2px dashed #e2e8f077; }
  .scene-box.hazard_bearing { animation: breathe 2.2s ease-in-out infinite; }
  .scene-box.at_risk { animation: heartbeat 0.9s ease-in-out infinite; }
  @keyframes breathe { 0%,100% { box-shadow:0 0 6px 1px #ef444488; } 50% { box-shadow:0 0 18px 5px #ef4444cc; } }
  @keyframes heartbeat { 0%,100% { box-shadow:0 0 6px 1px #f9731688; } 45% { box-shadow:0 0 16px 5px #f97316dd; } }
  .box-tag { position:absolute; top:-22px; left:-3px; font-size:11px; padding:2px 7px;
      border-radius:4px 4px 4px 0; color:white; white-space:nowrap; background:#475569;
      box-shadow:0 1px 3px rgba(0,0,0,.4); }
  .chip { position:absolute; background:#f59e0b; color:#231a00; font-size:11px; font-weight:bold;
      padding:3px 9px; border-radius:10px; transform:translateY(-120%);
      animation: chipin .3s ease-out; box-shadow:0 2px 6px rgba(0,0,0,.35); }
  .chip.docked { position:static; display:inline-block; margin:6px; transform:none; }
  @keyframes chipin { from { opacity:0; transform:translateY(-160%);} to { opacity:1; transform:translateY(-120%);} }
  .rail { background:transparent; border:none; padding:0; display:flex;
      flex-direction:column; gap:12px; }
  .station { padding:14px 18px; border-radius:14px; margin:0;
      border:1px solid var(--line); border-left:5px solid var(--line);
      background:var(--card); box-shadow:var(--shadow); opacity:.62; }
  .station.active, .station.done { opacity:1; }
  /* Per-stage tinted, shaded cards */
  .st-perceive.active, .st-perceive.done { background:linear-gradient(135deg,#eff6ff,#ffffff); border-left-color:#3b82f6; }
  .st-repair.active,   .st-repair.done   { background:linear-gradient(135deg,#fffbeb,#ffffff); border-left-color:#f59e0b; }
  .st-ground.active,   .st-ground.done   { background:linear-gradient(135deg,#f5f3ff,#ffffff); border-left-color:#8b5cf6; }
  .st-bind.active,     .st-bind.done     { background:linear-gradient(135deg,#ecfeff,#ffffff); border-left-color:#06b6d4; }
  .st-mask.active,     .st-mask.done     { background:linear-gradient(135deg,#fdf2f8,#ffffff); border-left-color:#ec4899; }
  .st-assemble.active, .st-assemble.done { background:linear-gradient(135deg,#f0fdf4,#ffffff); border-left-color:#16a34a; }
  .station.active { animation: stlift 1.2s ease-in-out infinite; box-shadow:var(--shadow-lift); }
  @keyframes stlift { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-1px); } }
  .station-name { font-weight:bold; letter-spacing:1.5px; font-size:14px; }
  .station-head { display:flex; align-items:baseline; gap:10px; margin-bottom:2px;
      cursor:pointer; list-style:none; }
  .station-head::-webkit-details-marker { display:none; }
  .station-secs { font-size:11px; color:var(--faint); }
  .station-compact { font-size:11px; color:var(--muted); margin-left:auto; }
  .station-chev { color:var(--faint); font-size:11px; transition:transform .2s; }
  details:not([open]) > .station-head .station-chev,
  details:not([open]) > .phase-head .station-chev { transform:rotate(-90deg); }
  .phase-card { background:var(--card); border:1px solid var(--line);
      border-radius:16px; padding:14px 16px; box-shadow:var(--shadow-lift); }
  .phase-head { display:flex; align-items:baseline; gap:10px; cursor:pointer;
      list-style:none; margin-bottom:10px; }
  .phase-head::-webkit-details-marker { display:none; }
  .phase-num { font-size:10px; font-weight:bold; letter-spacing:2px; color:white;
      background:var(--accent); border-radius:6px; padding:3px 8px; }
  .phase-title { font-weight:bold; letter-spacing:2px; font-size:14px; color:var(--ink); }
  .station-info, .station-loop { font-size:12px; color:var(--muted); }
  .station-detail { font-size:12px; color:#475569; margin-top:3px; line-height:1.6; }
  .station-detail.dim { color:var(--faint); }
  .station-loop { color:#d97706; margin-top:3px; }
  .rail-link { display:none; }
  .tickets { margin-top:14px; max-height:420px; overflow-y:auto; }
  .ticket { background:var(--card); border:1px solid var(--line); border-radius:12px;
      margin:10px 0; padding:11px 14px; box-shadow:var(--shadow); }
  .ticket.open { border-color:#f59e0b; animation: tkin .3s ease-out; }
  .ticket.fixed { opacity:.8; } .ticket.stood { border-color:#cbd5e1; }
  @keyframes tkin { from { opacity:0; transform:translateX(20px);} to { opacity:1; transform:none;} }
  .ticket summary { cursor:pointer; display:flex; gap:8px; align-items:center; list-style:none; }
  .ticket-kind { color:#d97706; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
  .ticket-label { font-family:monospace; font-size:12px; color:var(--ink); }
  .stamp { margin-left:auto; font-size:10px; font-weight:bold; letter-spacing:1px;
      padding:2px 8px; border-radius:6px; border:1px solid; }
  .stamp.open { color:#d97706; border-color:#f59e0b; background:#fffbeb; }
  .stamp.fixing { color:#b45309; border-color:#f59e0b; background:#fef3c7;
      animation: fixpulse 1s ease-in-out infinite; }
  @keyframes fixpulse { 50% { background:#fde68a; } }
  .ticket.fixing { border-color:#f59e0b; }
  /* Mini timeline inside each station card: nodes on a tinted connector. */
  .tl { margin-top:10px; position:relative; }
  .tl-row { display:flex; align-items:flex-start; gap:11px; position:relative;
      padding:4px 0; }
  .tl-row:not(:last-child)::after { content:""; position:absolute; left:9px;
      top:26px; bottom:-6px; width:2px; background:var(--tl-line, #cbd5e1);
      border-radius:1px; }
  .tl-node { width:20px; height:20px; border-radius:50%; border:2px solid #cbd5e1;
      background:#fff; color:#64748b; font-size:11px; font-weight:bold;
      display:flex; align-items:center; justify-content:center; flex:none;
      z-index:1; }
  .tl-node.ok { color:#16a34a; }
  .tl-node.warn { background:#fffbeb; }
  .tl-node.stood { color:#64748b; background:#f8fafc; }
  .tl-node.dim { color:#cbd5e1; border-style:dashed; }
  .tl-node.now { animation: nodepulse 1.1s ease-in-out infinite; }
  @keyframes nodepulse { 50% { transform:scale(1.18); } }
  .tl-text { font-size:12px; color:var(--muted); line-height:1.6; padding-top:2px; }
  .tl-text.dim { color:var(--faint); font-style:italic; }
  .tl-text.now { color:var(--ink); font-weight:600; }
  .tl-text.warn { color:#b45309; }
  .stamp.fixed { color:#16a34a; border-color:#86efac; background:#f0fdf4; }
  .stamp.stood { color:#64748b; border-color:#cbd5e1; background:#f8fafc; }
  .ticket-body { margin-top:8px; }
  .speaker { font-size:10px; letter-spacing:2px; color:var(--faint); }
  .bubble { background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:9px;
      font-family:monospace; font-size:12px; white-space:pre-wrap; color:#334155; }
  /* RAG result-diff button + panel */

  /* Control-plane toggles in the controls row (light) */
  .ctl-group{display:inline-flex; align-items:center; gap:7px;
             background:#ffffff; border:1px solid #cbd5e1; border-radius:9px;
             padding:5px 10px;}
  .ctl-lbl{font-size:10px; font-weight:800; letter-spacing:.06em;
           text-transform:uppercase; color:#64748b;}
  .ctl-toggle{display:inline-flex; gap:10px;}
  .ctl-toggle label{font-size:12px; font-weight:700; color:#334155;
                    display:inline-flex; align-items:center; gap:4px;
                    cursor:pointer;}
  .ctl-toggle input{accent-color:#2563eb; cursor:pointer;}

  /* Live status badge on each stage card header */
  .phase-status { margin-left:auto; margin-right:8px; font-size:11px;
                  font-weight:700; letter-spacing:.02em; }
  .phase-status.done { color:#16a34a; }
  .phase-status.active { color:#2563eb; animation: blink 1.2s infinite; }
  .phase-status.pending { color:#94a3b8; }

  /* WHERE-ARE-WE progress strip */
  .pstrip { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
            padding:10px 14px 8px; margin:6px 0 10px; }
  .pstrip-row { display:flex; align-items:center; gap:4px; flex-wrap:wrap; }
  .pchip { display:flex; align-items:center; gap:5px; }
  .pchip-dot { width:20px; height:20px; border-radius:50%; display:flex;
               align-items:center; justify-content:center; font-size:11px;
               font-weight:800; border:2px solid #cbd5e1; color:#94a3b8; }
  .pchip-dot.done { border-color:#16a34a; background:#16a34a; color:#fff; }
  .pchip-dot.active { border-color:#2563eb; color:#2563eb;
                      box-shadow:0 0 0 4px #2563eb22;
                      animation: blink 1.1s infinite; }
  .pchip-name { font-size:11px; font-weight:700; color:#94a3b8;
                letter-spacing:.02em; }
  .pchip-name.done { color:#334155; }
  .pchip-name.active { color:#2563eb; }
  .pstrip-link { width:16px; height:2px; background:#e2e8f0;
                 border-radius:2px; }
  .pstrip-link.done { background:#16a34a99; }
  .pstrip-now { margin-top:6px; font-size:12px; color:#2563eb;
                font-weight:600; }
  .pstrip-now.done { color:#16a34a; }
  .ribbon { position:absolute; top:10px; left:10px; z-index:6; background:#ffffffee;
      border:1px solid var(--line); border-radius:10px; padding:5px 12px; font-size:12px;
      letter-spacing:1px; color:#334155; box-shadow:var(--shadow); }
  .ribbon.busy::before { content:"● "; color:var(--accent); animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity:.2; } }
  .verdict-ribbon { z-index:7; }
  .verdict-ribbon.busy::after { content:" ●"; animation: blink 1s infinite; }
  .unc-toggle { font-size:12px; color:#475569; margin:2px 0 8px; display:flex; gap:14px; }
  .unc-toggle label { margin-right:10px; cursor:pointer; }
  .unc-panel { background:#f8fafc; border:1px solid var(--line); border-radius:10px;
      padding:8px 12px; margin:6px 0; }
  .unc-row { display:flex; align-items:baseline; gap:8px; margin:3px 0; }
  .unc-tag { font-size:11px; letter-spacing:.06em; text-transform:uppercase;
      color:#64748b; min-width:88px; }
  .unc-val { font-weight:700; color:#0f172a; }
  .unc-note { font-size:12px; color:#64748b; }
  .unc-why { font-size:12px; color:#475569; margin-top:6px; padding-top:6px;
      border-top:1px dashed var(--line); }
  .unc-gap { font-size:12px; font-weight:600; margin-top:4px; }
  .sweep { position:absolute; top:0; bottom:0; width:34%; left:-34%; z-index:4; pointer-events:none;
      background:linear-gradient(100deg, transparent 0%, #ffffff2a 50%, transparent 100%);
      animation: sweepmove 2.6s linear infinite; }
  @keyframes sweepmove { from { left:-34%; } to { left:110%; } }
  .scene-box.spotlight { animation: spotpulse .7s ease-in-out infinite !important; }
  @keyframes spotpulse { 0%,100% { box-shadow:0 0 8px 2px #f59e0baa; } 50% { box-shadow:0 0 22px 7px #f59e0b; } }
  .modal-backdrop { position:fixed; inset:0; background:rgba(15,23,42,.45); z-index:50;
      display:flex; align-items:center; justify-content:center; backdrop-filter:blur(2px); }
  .modal { background:var(--card); border:1px solid var(--line); border-radius:16px;
      padding:20px 22px; width:440px; box-shadow:var(--shadow-lift); }
  .insp-close { margin-left:auto; background:none; border:none; color:var(--faint);
      font-size:20px; cursor:pointer; }
  .insp-head { display:flex; gap:10px; align-items:center; }
  .insp-id { font-family:monospace; font-weight:bold; font-size:14px; color:var(--ink); }
  .insp-state { color:white; font-size:11px; padding:2px 8px; border-radius:10px; }
  .insp-desc { color:var(--muted); font-size:12px; margin:6px 0; }
  .insp-chain { font-size:12px; color:#334155; }
  .scrub { margin-top:10px; }
  .agent-dock { background:var(--card); border:1px solid var(--line); border-radius:16px;
      padding:14px 16px; margin-top:12px; box-shadow:var(--shadow); }
  .agent-head { display:flex; align-items:baseline; gap:12px; margin-bottom:8px; }
  .agent-title { font-weight:bold; letter-spacing:2px; font-size:13px; color:var(--accent); }
  .agent-sub { font-size:11px; color:var(--faint); }
  .agent-transcript { max-height:280px; overflow-y:auto; display:flex;
      flex-direction:column; gap:8px; margin-bottom:10px; }
  .bubble-user { align-self:flex-end; background:var(--accent); color:white;
      border-radius:14px 14px 4px 14px; padding:8px 13px; font-size:13px; max-width:70%; }
  .bubble-agent { align-self:flex-start; background:#f1f5f9; border:1px solid var(--line);
      border-radius:14px 14px 14px 4px; padding:9px 13px; font-size:13px; max-width:85%; }
  .bubble-agent-text { white-space:pre-wrap; }
  .tool-chips { margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  .tool-chip { font-family:monospace; font-size:10px; background:#e0e7ff; color:#3730a3;
      border-radius:8px; padding:2px 8px; }
  .agent-steps { --tl-line:#94a3b855; margin-bottom:8px; }
  .step-tool { font-family:monospace; font-size:11px; background:#e0e7ff; color:#3730a3;
      border-radius:6px; padding:1px 7px; }
  .bubble-agent-text.working { color:var(--faint); animation: blink 1s infinite; }
  .agent-inputrow { display:flex; gap:8px; }
  .agent-inputbox { flex:1; background:#fff; border:1px solid var(--line); color:var(--ink);
      border-radius:10px; padding:9px 12px; }
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

app.layout = html.Div([
    html.H1("CEE+ AGENTIC · STAGES 1-2 · PERCEPTION → ASSESSMENT"),
    html.Div([
        dcc.Upload(id="upload", children=html.Div("drop / pick image"),
                   className="upload", multiple=False),
        dcc.Input(id="caption", type="text",
                  placeholder="caption (part of the input, like the corpus)"),
        html.Button("ANALYZE", id="analyze", className="go", n_clicks=0),
        dcc.Dropdown(id="replay", options=saved_records(),
                     placeholder="or replay a saved run...",
                     style={"width": "260px", "color": "#111"}),
        # Control-plane toggles (no restart, no env vars). Applied at the
        # moment a run launches. Defaults from 2026-07-28 (Sunny): LangGraph
        # and both. The control we ship should be the one every run exercises,
        # and a RAG seam only measured when someone remembers to switch it on
        # is a seam nobody measures. Either can still be switched per run.
        html.Div([
            html.Span("control", className="ctl-lbl"),
            dcc.RadioItems(
                id="control-mode", value="langgraph", inline=True,
                options=[{"label": "Python", "value": "python"},
                         {"label": "LangGraph", "value": "langgraph"}],
                className="ctl-toggle"),
        ], className="ctl-group"),
        html.Div([
            html.Span("rule lookup", className="ctl-lbl"),
            dcc.RadioItems(
                id="retrieval-mode", value="both", inline=True,
                options=[{"label": "exact", "value": "rulebook"},
                         {"label": "RAG", "value": "rag"},
                         {"label": "both", "value": "both"}],
                className="ctl-toggle"),
        ], className="ctl-group"),
    ], className="controls"),
    html.Div(id="instruments"),
    html.Div([
        html.Div([html.Div(id="scene", className="scene"),
                  html.Div(dcc.Slider(id="scrub", min=0, max=1, step=1, value=1,
                                      marks=None, updatemode="drag",
                                      tooltip={"placement": "bottom"}),
                           className="scrub"),
                  # The conversation agent, docked under the scene.
                  html.Div([
                      html.Div([html.Span("ASK THE ANALYST", className="agent-title"),
                                html.Span("answers come only from the run records",
                                          className="agent-sub")],
                               className="agent-head"),
                      html.Div(id="agent-transcript", className="agent-transcript"),
                      html.Div([
                          dcc.Input(id="agent-input", type="text", debounce=True,
                                    placeholder="ask about this run… "
                                                "(why is child_1 at risk?)",
                                    className="agent-inputbox", n_submit=0),
                          html.Button("ASK", id="agent-send", className="go",
                                      n_clicks=0),
                      ], className="agent-inputrow"),
                  ], className="agent-dock", id="agent-dock")]),
        # Phase cards stack CHRONOLOGICALLY, pipeline order top-down (the
        # rail reads like the pipeline diagram; feeds scroll, flight plans
        # don't reorder). A finished phase auto-collapses once the next
        # one starts; the user can always re-open it.
        html.Div([
            html.Details([
                html.Summary([html.Span("STAGE 1", className="phase-num"),
                              html.Span("PERCEPTION", className="phase-title"),
                              html.Span(id="phase0-status"),
                              html.Span("▾", className="station-chev")],
                             className="phase-head"),
                html.Div(id="rail"), html.Div(id="tickets"),
            ], className="phase-card", open=True, id="phase0-card"),
            html.Details([
                html.Summary([html.Span("STAGE 2", className="phase-num"),
                              html.Span("SCENE ASSESSMENT",
                                        className="phase-title"),
                              html.Span(id="phase1-status"),
                              html.Span("▾", className="station-chev")],
                             className="phase-head"),
                # The uncertainty-view toggle (Sunny: option to show the
                # model's self-reported number, our measured one, or both).
                dcc.RadioItems(
                    id="unc-view", value="both", inline=True,
                    options=[{"label": "self-reported", "value": "self"},
                             {"label": "measured", "value": "measured"},
                             {"label": "both (calibration)", "value": "both"}],
                    className="unc-toggle"),
                html.Div(id="assess-body"),
            ], className="phase-card", open=True, id="phase1-card"),
            html.Details([
                html.Summary([html.Span("STAGE 4", className="phase-num"),
                              html.Span("RECOMMENDATION",
                                        className="phase-title"),
                              html.Span(id="phase2-status"),
                              html.Span("▾", className="station-chev")],
                             className="phase-head"),
                html.Div(id="stage4-body"),
            ], className="phase-card", open=True, id="phase2-card"),
        ]),
    ], className="main"),
    html.Div(id="inspector-modal"),
    dcc.Store(id="run-id"), dcc.Store(id="selected"),
    dcc.Store(id="upload-cache"), dcc.Store(id="render-key"),
    dcc.Interval(id="tick", interval=700),
], className="wrap")


@app.callback(Output("upload-cache", "data"), Output("upload", "children"),
              Output("caption", "value"),
              Input("upload", "contents"), State("upload", "filename"),
              prevent_initial_call=True)
def cache_upload(contents, filename):
    """Cache the pick AND show it: the upload control becomes a thumbnail
    plus filename, so there is never doubt about which image is loaded.
    The caption CLEARS on a new image: a stale caption from the previous
    scene contaminates perception (the A_fire run that inherited
    C_tanker's caption and honestly ticketed a missing tanker_truck)."""
    picked = html.Div([
        html.Img(src=contents, className="upload-thumb"),
        html.Span(filename or "image", className="upload-name"),
    ], className="upload-inner")
    return {"contents": contents, "filename": filename}, picked, ""


@app.callback(Output("run-id", "data"),
              Input("analyze", "n_clicks"), Input("replay", "value"),
              State("upload-cache", "data"), State("caption", "value"),
              State("control-mode", "value"),
              State("retrieval-mode", "value"),
              prevent_initial_call=True)
def start_run(_clicks, replay_path, cached, caption,
              control_mode, retrieval_choice):
    # Apply the on-screen toggles for this run (in-process override).
    from agentic.graph_live import set_control
    from agentic.retrieval import set_retrieval
    set_control(control_mode)
    set_retrieval(retrieval_choice)
    if ctx.triggered_id == "replay" and replay_path:
        return start_replay(replay_path)
    if ctx.triggered_id == "analyze" and cached and cached.get("contents"):
        header, b64 = cached["contents"].split(",", 1)
        return start_live_run(base64.b64decode(b64),
                              cached.get("filename") or "scene.jpg",
                              caption or "")
    return dash.no_update


# Per-page-load agent transcripts, keyed by thread id. Each turn:
# {"q", "steps": [trajectory events], "a": answer|None, "pending": bool}.
AGENT_LOGS: dict[str, list[dict[str, Any]]] = {}


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{v}" for v in args.values())


def agent_transcript_component(log: list[dict[str, Any]]) -> list[Any]:
    """Render the transcript, INCLUDING each turn's trajectory: the steps
    the agent took (thinking, tool calls, tool results) shown as a mini
    timeline above the answer, live-pulsing while the turn is in flight.
    Tool use is visible before any answer exists."""
    bubbles: list[Any] = []
    for turn in log:
        bubbles.append(html.Div(turn["q"], className="bubble-user"))
        rows: list[Any] = []
        steps = turn.get("steps", [])
        for i, s in enumerate(steps):
            live = turn.get("pending") and i == len(steps) - 1
            kind = s.get("step")
            if kind == "thinking":
                glyph, mod, text = "…", "ask", "thinking..."
            elif kind == "tool_call":
                glyph, mod = "→", "info"
                text = html.Span(f"{s['tool']}({_fmt_args(s.get('args', {}))})",
                                 className="step-tool")
            elif kind == "tool_result":
                glyph = "✓" if s.get("ok") else "!"
                mod = "ok" if s.get("ok") else "warn"
                text = f"{s['tool']}: {s.get('summary', '')}"
            elif kind == "answer":
                continue                      # the answer renders as the bubble
            else:
                glyph, mod, text = "·", "info", str(s)
            node_style = {"boxShadow": "0 0 0 4px #2563eb22"} if live else {}
            rows.append(html.Div([
                html.Div(glyph, className=f"tl-node {mod}" + (" now" if live else ""),
                         style=node_style),
                html.Div(text, className="tl-text" + (" now" if live else "")),
            ], className="tl-row"))
        body: list[Any] = []
        if rows:
            body.append(html.Div(rows, className="tl agent-steps"))
        if turn.get("pending") and turn.get("a") is None:
            body.append(html.Div("…", className="bubble-agent-text working"))
        elif turn.get("a") is not None:
            body.append(html.Div(turn["a"], className="bubble-agent-text"))
            if turn.get("unverified"):
                # Deterministic groundedness flag: these entity ids appear
                # in the answer but in NO tool result this turn. We badge,
                # we never block: show the failure, don't hide it.
                body.append(html.Div(
                    "⚠ not in retrieved evidence: " + ", ".join(turn["unverified"]),
                    className="bubble-agent-text unverified-badge",
                    style={"color": "#b45309", "background": "#fffbeb",
                           "border": "1px solid #fcd34d", "borderRadius": "8px",
                           "padding": "4px 10px", "marginTop": "6px",
                           "fontSize": "12px", "fontWeight": "600"}))
        bubbles.append(html.Div(body, className="bubble-agent"))
    return bubbles


@app.callback(Output("agent-input", "value"),
              Input("agent-send", "n_clicks"), Input("agent-input", "n_submit"),
              State("agent-input", "value"), State("run-id", "data"),
              prevent_initial_call=True)
def ask_agent(_clicks, _submit, question, run_id):
    """Start one dialogue turn in a background thread. The transcript
    (with the live trajectory) renders via the tick interval, exactly like
    the pipeline's own event stream."""
    if not question or not question.strip():
        return dash.no_update
    run = RUNS.get(run_id or "")
    focus = run.get("record_name") if run else None
    thread = f"ui-{run_id or 'global'}"
    log = AGENT_LOGS.setdefault(thread, [])
    if log and log[-1].get("pending"):
        # One dialogue turn at a time per conversation: respond() also
        # serializes on a per-thread lock, but queueing here would still
        # answer questions out of the order the user sees. Keep the text
        # in the box; they can resend when the turn completes.
        return dash.no_update
    turn = {"q": question.strip(), "steps": [], "a": None, "pending": True}
    log.append(turn)

    def worker() -> None:
        try:
            from agentic.dialogue import respond
            out = respond(thread, turn["q"], focus_run=focus,
                          on_step=turn["steps"].append)
            turn["a"] = out["answer"]
            turn["unverified"] = out.get("unverified_ids") or []
        except Exception as exc:
            turn["a"] = (f"agent unavailable: {exc}. Is Ollama running with "
                         f"the dialogue model pulled (ollama pull qwen2.5:7b)?")
        finally:
            turn["pending"] = False

    threading.Thread(target=worker, daemon=True).start()
    return ""


@app.callback(Output("agent-dock", "n_clicks", allow_duplicate=True),
              Input({"type": "judge-btn", "n": dash.dependencies.ALL},
                    "n_clicks"),
              State("run-id", "data"), prevent_initial_call=True)
def run_judge(clicks, run_id):
    """Blind pairwise judging on demand: pre (canonical) vs post (final)
    verdicts, shown unlabeled to the JUDGE model (different training
    family; JUDGE_MODEL env, default llama3.1:8b)."""
    if not clicks or not any(c for c in clicks if c):
        return dash.no_update
    run = RUNS.get(run_id or "")
    if not run or run.get("judge", {}).get("status") == "running":
        return dash.no_update
    d = derive(run["events"])
    hist = d["assess"].get("verdict_history") or []
    if len(hist) < 1 or not d.get("result"):
        return dash.no_update
    pre = _verdict_to_assessment(hist[0])
    post = _verdict_to_assessment(hist[-1])
    run["judge"] = {"status": "running"}

    def worker() -> None:
        try:
            from agentic.evals import judge_pairwise
            run["judge"] = judge_pairwise(pre, post, d["result"],
                                          run.get("record_name") or "scene")
        except Exception as exc:
            run["judge"] = {"winner": "error", "raw": str(exc)}

    threading.Thread(target=worker, daemon=True).start()
    return dash.no_update


@app.callback(Output("selected", "data"),
              Input({"type": "scene-box", "oid": ALL}, "n_clicks"),
              Input({"type": "insp-close", "n": ALL}, "n_clicks"),
              prevent_initial_call=True)
def select_entity(box_clicks, close_clicks):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") == "insp-close":
        return None
    if isinstance(trig, dict) and trig.get("type") == "scene-box" and any(box_clicks):
        return trig.get("oid")
    return dash.no_update


@app.callback(
    Output("rail", "children"), Output("tickets", "children"),
    Output("instruments", "children"), Output("scene", "children"),
    Output("inspector-modal", "children"),
    Output("scrub", "max"), Output("scrub", "value"),
    Output("agent-transcript", "children"),
    Output("assess-body", "children"), Output("stage4-body", "children"),
    Output("phase0-card", "open"),
    Output("phase0-status", "children"), Output("phase1-status", "children"),
    Output("phase2-status", "children"),
    Output("render-key", "data"),
    Input("tick", "n_intervals"), Input("scrub", "value"),
    Input("unc-view", "value"),
    State("run-id", "data"), State("selected", "data"), State("scrub", "max"),
    State("upload-cache", "data"), State("render-key", "data"))
def render(_n, scrub_value, unc_view, run_id, selected, scrub_max, cached,
           prev_key):
    thread = f"ui-{run_id or 'global'}"
    log = AGENT_LOGS.get(thread, [])
    # Agent fingerprint: any new step or completed answer re-renders.
    agent_sig = [[len(t.get("steps", [])), t.get("a") is not None] for t in log]
    run = RUNS.get(run_id or "")
    if not run:
        empty = derive([])
        key = ["empty", bool(cached and cached.get("contents")), agent_sig,
               unc_view, "pending"]
        if key == prev_key:
            return (dash.no_update,) * 15
        # A picked-but-unanalyzed image previews in the scene immediately,
        # with a ribbon saying it's ready.
        if cached and cached.get("contents"):
            scene = [html.Img(src=cached["contents"], className="scene-img"),
                     html.Div(f"{cached.get('filename', 'image')} — ready, "
                              f"press ANALYZE", className="ribbon")]
        else:
            scene = scene_component(empty, None)
        return (rail_component(empty), tickets_component(empty),
                instruments_component(empty), scene,
                inspector_component(empty, None), 1, 1,
                agent_transcript_component(log),
                assess_component(empty, unc_view), stage4_component(empty, None),
                dash.no_update,
                phase_status_span([]), phase_status_span([]),
                html.Span("○ waiting", className="phase-status pending"), key)
    events = run["events"]
    total = max(1, len(events))
    # Follow-live rule: the slider sticks to the right edge unless the user
    # dragged it left; dragging back to the edge resumes following.
    following = scrub_value is None or scrub_max is None or scrub_value >= scrub_max
    k = total if (following or ctx.triggered_id != "scrub") and following else min(scrub_value, total)
    d = derive(events[:k])
    assess_status = d["assess"]["status"]
    # Render cache: when nothing changed since the last tick, leave the DOM
    # alone. This is what preserves the user's collapse/expand toggles
    # (re-rendering identical children would reset every <details>).
    key = [run_id, total, k, selected, agent_sig, unc_view, assess_status,
           (run.get("judge") or {}).get("winner",
                                        (run.get("judge") or {}).get("status"))]
    if key == prev_key and ctx.triggered_id != "scrub":
        return (dash.no_update,) * 15
    # Auto-collapse Phase 0 exactly ONCE, at the moment Phase 1 starts
    # (comparing against prev_key keeps later renders from stomping the
    # user's manual re-open).
    prev_status = prev_key[6] if (isinstance(prev_key, list)
                                  and len(prev_key) > 6) else "pending"
    phase0_open = (False if (assess_status != "pending"
                             and prev_status == "pending")
                   else dash.no_update)
    judge = run.get("judge")
    # Live per-card badges (Sunny prefers the status ON the stage cards,
    # not a top bar): "● <what it's doing> · step 3/6" while running.
    s1_steps, s2_steps = pipeline_steps(d)
    now = (d.get("activity") or {}).get("text", "")
    p0_badge = phase_status_span(
        s1_steps, now if any(s == "active" for _, s in s1_steps) else "")
    # Stage 2 is SETTLED once a petition has concluded: either the second look
    # added nothing (stage 2 never re-runs, so no further events are coming) or
    # the outcome has been recorded. Either way the badge must stop pulsing.
    _p = ((d.get("assess") or {}).get("petition") or {})
    _settled = bool(_p) and _p.get("status") != "in_flight"
    p1_badge = phase_status_span(
        s2_steps, now if any(s == "active" for _, s in s2_steps) else "",
        settled=_settled)
    rail_view: Any = rail_component(d)
    assess_view: Any = assess_component(d, unc_view,
                                        run.get("record_name"), judge)
    e0 = d.get("epoch0")
    if e0:
        # Sunny 2026-07-22: never overwrite the first pass — keep it as a
        # collapsed BEFORE PETITION panel so the comparison is on screen.
        # A stage-2 petition snapshots ONLY the assessment (the image was
        # never re-looked, so the perception panels stay live) — its e0
        # has no "stages" key, and the rail must not touch it.
        if "stages" in e0:
            rail_view = html.Div([
                html.Details([
                    html.Summary("BEFORE PETITION — original perception",
                                 className="unc-tag",
                                 style={"cursor": "pointer",
                                        "padding": "4px 0"}),
                    rail_component(e0), tickets_component(e0),
                ], open=False,
                    style={"opacity": "0.85", "borderBottom":
                           "2px dashed #c4b5fd", "marginBottom": "8px",
                           "paddingBottom": "6px"}),
                html.Div("RE-PERCEPTION (after petition)",
                         className="unc-tag",
                         style={"color": "#7c3aed", "margin": "2px 0"}),
                rail_view])
        stage2 = "stages" not in e0
        assess_view = [
            html.Details([
                html.Summary("BEFORE PETITION — original assessment",
                             className="unc-tag",
                             style={"cursor": "pointer",
                                    "padding": "4px 0"}),
                html.Div(assess_component(e0, unc_view)),
            ], open=False,
                style={"opacity": "0.85", "borderBottom":
                       "2px dashed #c4b5fd", "marginBottom": "8px",
                       "paddingBottom": "6px"}),
            html.Div("FRESH RE-ASK (same stage, after petition)"
                     if stage2 else "RE-ASSESSMENT (after petition)",
                     className="unc-tag",
                     style={"color": "#7c3aed", "margin": "2px 0"}),
        ] + assess_view
    return (rail_view, tickets_component(d), instruments_component(d),
            scene_component(d, run.get("image_src")),
            inspector_component(d, selected),
            total, total if following else k,
            agent_transcript_component(log),
            assess_view, stage4_component(d, run.get("image_src")),
            phase0_open, p0_badge, p1_badge, stage4_status_span(d), key)


if __name__ == "__main__":
    # Debug mode ON by default: Dash then hot-reloads the browser tab when
    # the server code changes. Without it, a tab from before a restart
    # keeps posting the OLD callback wiring and every interval tick 500s
    # (the KeyError / IndexError storms Sunny hit on 2026-07-21). Disable
    # with AGENTIC_UI_DEBUG=0 for a demo.
    import os as _os
    debug = _os.getenv("AGENTIC_UI_DEBUG", "1") != "0"
    app.run(debug=debug, port=8060)
