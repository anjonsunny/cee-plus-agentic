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

SCENES_DIR = REPO_ROOT / "experiments" / "agentic_scenes"
PERCEPTION_DIR = SCENES_DIR / "perception"
STAGES = ["Perceive", "Repair", "Ground", "Bind", "Mask", "Assemble"]

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
                          if k not in ("type", "stage", "seconds", "entities")}
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
        if t == "stage_started":
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
    }
    sink = make_event_sink(run_id, run_dir)

    def worker() -> None:
        try:
            from agentic.perception import run_perception
            run_perception(image_path, caption=caption, out_dir=run_dir,
                           on_event=sink)
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
                        "done": True, "error": None}
        return run_id
    record = json.loads(path.read_text())
    RUNS[run_id] = {
        "events": build_replay_events(record),
        "image_src": image_src_for_record(path),
        "done": True, "error": None,
    }
    return run_id


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
    return html.Div(items, className="rail")


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
        label = ("no violations — clean on arrival"
                 if d["stop_reason"] == "clean" and not d["violations"]
                 else "waiting for the model's first answer...")
        cards = [html.Div(label, className="ticket-empty")]
    return html.Div(cards, className="tickets")


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

    return html.Div([
        tile(n_entities, "entities"),
        tile(open_v, "open", "amber" if open_v else ""),
        tile(fixed_v, "fixed", "green" if fixed_v else ""),
        tile(stood_v, "stood", "gray" if stood_v else ""),
        tile("●" * d["rounds_used"] + "○" * max(0, 2 - d["rounds_used"]), "rounds"),
        tile(f"{matched}/{matched + fallback}" if (matched + fallback) else "–", "bound"),
    ], className="instruments")


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
app = dash.Dash(__name__, title="CEE+ Agentic — Perception",
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
  .scene { position:relative; border-radius:14px; overflow:hidden; background:#0f172a;
      box-shadow:var(--shadow-lift); }
  .scene-img { width:100%; display:block; }
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
  .ribbon { position:absolute; top:10px; left:10px; z-index:6; background:#ffffffee;
      border:1px solid var(--line); border-radius:10px; padding:5px 12px; font-size:12px;
      letter-spacing:1px; color:#334155; box-shadow:var(--shadow); }
  .ribbon.busy::before { content:"● "; color:var(--accent); animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity:.2; } }
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
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

app.layout = html.Div([
    html.H1("CEE+ AGENTIC · STAGE 1 PERCEPTION"),
    html.Div([
        dcc.Upload(id="upload", children=html.Div("drop / pick image"),
                   className="upload", multiple=False),
        dcc.Input(id="caption", type="text",
                  placeholder="caption (part of the input, like the corpus)"),
        html.Button("ANALYZE", id="analyze", className="go", n_clicks=0),
        dcc.Dropdown(id="replay", options=saved_records(),
                     placeholder="or replay a saved run...",
                     style={"width": "260px", "color": "#111"}),
    ], className="controls"),
    html.Div(id="instruments"),
    html.Div([
        html.Div([html.Div(id="scene", className="scene"),
                  html.Div(dcc.Slider(id="scrub", min=0, max=1, step=1, value=1,
                                      marks=None, updatemode="drag",
                                      tooltip={"placement": "bottom"}),
                           className="scrub"),
                  # Reserved: the conversation agent docks here next.
                  html.Div(id="agent-dock")]),
        # PHASE 0 · PERCEPTION: the whole live panel for this stage lives
        # under one collapsible phase card. Later stages (threats, graphs,
        # intervention) will stack their own phase cards below it.
        html.Div(html.Details([
            html.Summary([html.Span("PHASE 0", className="phase-num"),
                          html.Span("PERCEPTION", className="phase-title"),
                          html.Span("▾", className="station-chev")],
                         className="phase-head"),
            html.Div(id="rail"), html.Div(id="tickets"),
        ], className="phase-card", open=True)),
    ], className="main"),
    html.Div(id="inspector-modal"),
    dcc.Store(id="run-id"), dcc.Store(id="selected"),
    dcc.Store(id="upload-cache"), dcc.Store(id="render-key"),
    dcc.Interval(id="tick", interval=700),
], className="wrap")


@app.callback(Output("upload-cache", "data"), Output("upload", "children"),
              Input("upload", "contents"), State("upload", "filename"),
              prevent_initial_call=True)
def cache_upload(contents, filename):
    """Cache the pick AND show it: the upload control becomes a thumbnail
    plus filename, so there is never doubt about which image is loaded."""
    picked = html.Div([
        html.Img(src=contents, className="upload-thumb"),
        html.Span(filename or "image", className="upload-name"),
    ], className="upload-inner")
    return {"contents": contents, "filename": filename}, picked


@app.callback(Output("run-id", "data"),
              Input("analyze", "n_clicks"), Input("replay", "value"),
              State("upload-cache", "data"), State("caption", "value"),
              prevent_initial_call=True)
def start_run(_clicks, replay_path, cached, caption):
    if ctx.triggered_id == "replay" and replay_path:
        return start_replay(replay_path)
    if ctx.triggered_id == "analyze" and cached and cached.get("contents"):
        header, b64 = cached["contents"].split(",", 1)
        return start_live_run(base64.b64decode(b64),
                              cached.get("filename") or "scene.jpg",
                              caption or "")
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
    Output("render-key", "data"),
    Input("tick", "n_intervals"), Input("scrub", "value"),
    State("run-id", "data"), State("selected", "data"), State("scrub", "max"),
    State("upload-cache", "data"), State("render-key", "data"))
def render(_n, scrub_value, run_id, selected, scrub_max, cached, prev_key):
    run = RUNS.get(run_id or "")
    if not run:
        empty = derive([])
        key = ["empty", bool(cached and cached.get("contents"))]
        if key == prev_key:
            return (dash.no_update,) * 7 + (dash.no_update,)
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
                inspector_component(empty, None), 1, 1, key)
    events = run["events"]
    total = max(1, len(events))
    # Follow-live rule: the slider sticks to the right edge unless the user
    # dragged it left; dragging back to the edge resumes following.
    following = scrub_value is None or scrub_max is None or scrub_value >= scrub_max
    k = total if (following or ctx.triggered_id != "scrub") and following else min(scrub_value, total)
    # Render cache: when nothing changed since the last tick, leave the DOM
    # alone. This is what preserves the user's collapse/expand toggles
    # (re-rendering identical children would reset every <details>).
    key = [run_id, total, k, selected]
    if key == prev_key and ctx.triggered_id != "scrub":
        return (dash.no_update,) * 7 + (dash.no_update,)
    d = derive(events[:k])
    return (rail_component(d), tickets_component(d), instruments_component(d),
            scene_component(d, run.get("image_src")),
            inspector_component(d, selected),
            total, total if following else k, key)


if __name__ == "__main__":
    # Debug mode ON by default: Dash then hot-reloads the browser tab when
    # the server code changes. Without it, a tab from before a restart
    # keeps posting the OLD callback wiring and every interval tick 500s
    # (the KeyError / IndexError storms Sunny hit on 2026-07-21). Disable
    # with AGENTIC_UI_DEBUG=0 for a demo.
    import os as _os
    debug = _os.getenv("AGENTIC_UI_DEBUG", "1") != "0"
    app.run(debug=debug, port=8060)
