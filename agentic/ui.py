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

SCENES_DIR = REPO_ROOT / "experiments" / "agentic_scenes"
PERCEPTION_DIR = SCENES_DIR / "perception"
STAGES = ["Perceive", "Repair", "Ground", "Bind", "Mask", "Assemble"]

STATE_KIND_CSS = {
    "hazard_bearing": "#ef4444",
    "at_risk": "#f97316",
    "normal": "#3b82f6",
    "unknown": "#94a3b8",
}

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
    ev.append({"type": "stage_done", "stage": "Perceive", "seconds": None,
               "n_entities": len(objs)})
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
    if not PERCEPTION_DIR.exists():
        return []
    return [{"label": p.name.replace("__perception.json", ""), "value": str(p)}
            for p in sorted(PERCEPTION_DIR.glob("*__perception.json"))]


def image_src_for_record(json_path: Path) -> str | None:
    stem = json_path.name.replace("__perception.json", "")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = SCENES_DIR / f"{stem}{ext}"
        if p.exists():
            mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


# ── Live run (background thread) ────────────────────────────────────────


def start_live_run(image_bytes: bytes, filename: str, caption: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    tmp = Path("/tmp") / f"agentic_ui_{run_id}_{filename or 'scene.jpg'}"
    tmp.write_bytes(image_bytes)
    mime = mimetypes.guess_type(str(tmp))[0] or "image/jpeg"
    RUNS[run_id] = {
        "events": [],
        "image_src": f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}",
        "done": False, "error": None,
    }

    def worker() -> None:
        try:
            from agentic.perception import run_perception
            run_perception(tmp, caption=caption,
                           out_dir=tmp.parent / f"agentic_ui_{run_id}_out",
                           on_event=RUNS[run_id]["events"].append)
        except Exception as exc:  # surfaced as a card, never a dead screen
            RUNS[run_id]["events"].append(
                {"type": "run_error", "message": str(exc)})
            RUNS[run_id]["error"] = str(exc)
        finally:
            RUNS[run_id]["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return run_id


def start_replay(json_path: str) -> str:
    run_id = uuid.uuid4().hex[:8]
    record = json.loads(Path(json_path).read_text())
    RUNS[run_id] = {
        "events": build_replay_events(record),
        "image_src": image_src_for_record(Path(json_path)),
        "done": True, "error": None,
    }
    return run_id


# ── Components (each a pure function of the derived state) ──────────────


def rail_component(d: dict[str, Any]) -> html.Div:
    """The stage rail. Repair renders as a loop station with round pips."""
    items = []
    for s in STAGES:
        st = d["stages"][s]
        cls = {"pending": "station", "active": "station active",
               "done": "station done"}[st["status"]]
        secs = f"{st['seconds']:.1f}s" if st.get("seconds") else ""
        body = [html.Div(s, className="station-name"),
                html.Div(secs or st.get("info", ""), className="station-info")]
        if s == "Repair":
            pips = "".join("●" if i < d["rounds_used"] else "○" for i in range(2))
            stamp = {"clean": "CLEAN", "no_change": "STOOD GROUND",
                     "cap_reached": "CAP", "skipped": "SKIPPED", None: ""}.get(
                d["stop_reason"], "")
            body.append(html.Div(f"loop {pips}  {stamp}", className="station-loop"))
        items.append(html.Div(body, className=cls))
        items.append(html.Div("│", className="rail-link"))
    return html.Div(items[:-1], className="rail")


def tickets_component(d: dict[str, Any]) -> html.Div:
    """Violation tickets with lifecycle stamps; expanded = the Rulebook's
    exact instruction (the Model side lives in the round summary)."""
    cards = []
    for key, v in d["violations"].items():
        status = v["status"]
        stamp = {"open": ("OPEN", "stamp open"),
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


def _pct_box(bbox: list[int], size: list[int]) -> dict[str, str]:
    w, h = size
    x1, y1, x2, y2 = bbox
    return {"left": f"{100 * x1 / w:.2f}%", "top": f"{100 * y1 / h:.2f}%",
            "width": f"{100 * (x2 - x1) / w:.2f}%",
            "height": f"{100 * (y2 - y1) / h:.2f}%"}


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
    final_by_id = {o["object_id"]: o for o in
                   (d["result"] or {}).get("detected_objects", [])}
    if size:
        for a in d["anchors"]:
            oid = a.get("object_id")
            bound = d["bound"].get(oid)
            final = final_by_id.get(oid)
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
                    className=f"scene-box {final['state_kind']}", style=style, n_clicks=0))
            elif bound and bound.get("bbox"):
                style = _pct_box(bound["bbox"], size)
                children.append(html.Div(
                    html.Span(oid, className="box-tag"),
                    id={"type": "scene-box", "oid": oid},
                    className="scene-box bound", style=style, n_clicks=0))
            elif a.get("anchor_bbox"):
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
                if box:
                    style = _pct_box(box, size)
                    children.append(html.Div(chip_txt, className="chip",
                                             style={"left": style["left"], "top": style["top"]}))
                    continue
            children.append(html.Div(chip_txt, className="chip docked"))
    return children


def inspector_component(d: dict[str, Any], selected: str | None) -> html.Div:
    result = d["result"]
    if not selected or not result:
        return html.Div("click a box to inspect an entity", className="inspector-empty")
    obj = next((o for o in result["detected_objects"] if o["object_id"] == selected), None)
    if not obj:
        return html.Div("entity not found", className="inspector-empty")
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
    return html.Div([
        html.Div([html.Span(obj["object_id"], className="insp-id"),
                  html.Span(obj["state"], className="insp-state",
                            style={"background": color})], className="insp-head"),
        html.Div(obj.get("description", ""), className="insp-desc"),
        html.Ul([html.Li(c) for c in chain], className="insp-chain"),
    ], className="inspector")


# ── App assembly ────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="CEE+ Agentic — Perception")

app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  :root { color-scheme: dark; }
  body { background:#0b1220; color:#e2e8f0; font-family:'Helvetica Neue',Arial,sans-serif; margin:0; }
  .wrap { max-width:1500px; margin:0 auto; padding:14px 20px; }
  h1 { font-size:17px; letter-spacing:2px; color:#93c5fd; }
  .controls { display:flex; gap:10px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
  .controls .upload { border:1px dashed #334155; border-radius:8px; padding:8px 14px; cursor:pointer; }
  .controls input[type=text] { background:#111a2e; border:1px solid #334155; color:#e2e8f0;
      border-radius:8px; padding:8px 10px; width:420px; }
  .go { background:#1d4ed8; color:white; border:none; border-radius:8px; padding:9px 18px;
      font-weight:bold; letter-spacing:1px; cursor:pointer; }
  .instruments { display:flex; gap:8px; margin:8px 0; }
  .tile { background:#111a2e; border:1px solid #1e293b; border-radius:10px; padding:6px 14px;
      min-width:70px; text-align:center; }
  .tile-value { font-size:18px; font-weight:bold; }
  .tile-label { font-size:10px; color:#64748b; letter-spacing:1px; text-transform:uppercase; }
  .tile.amber .tile-value { color:#f59e0b; } .tile.green .tile-value { color:#22c55e; }
  .tile.gray .tile-value { color:#94a3b8; }
  .main { display:grid; grid-template-columns: 1fr 420px; gap:14px; }
  .scene { position:relative; border-radius:12px; overflow:hidden; background:#000; }
  .scene-img { width:100%; display:block; }
  .scene-empty, .inspector-empty, .ticket-empty { color:#475569; padding:30px; text-align:center; }
  .scene-box { position:absolute; border:3px solid #94a3b8; border-radius:4px; cursor:pointer; }
  .scene-box.anchor { border:2px dashed #64748b55; }
  .scene-box.hazard_bearing { animation: breathe 2.2s ease-in-out infinite; }
  .scene-box.at_risk { animation: heartbeat 0.9s ease-in-out infinite; }
  @keyframes breathe { 0%,100% { box-shadow:0 0 6px 1px #ef444488; } 50% { box-shadow:0 0 18px 5px #ef4444cc; } }
  @keyframes heartbeat { 0%,100% { box-shadow:0 0 6px 1px #f9731688; } 45% { box-shadow:0 0 16px 5px #f97316dd; } }
  .box-tag { position:absolute; top:-22px; left:-3px; font-size:11px; padding:2px 7px;
      border-radius:4px 4px 4px 0; color:white; white-space:nowrap; background:#475569; }
  .chip { position:absolute; background:#f59e0bee; color:#111; font-size:11px; font-weight:bold;
      padding:3px 8px; border-radius:10px; transform:translateY(-120%); animation: chipin .3s ease-out; }
  .chip.docked { position:static; display:inline-block; margin:6px; transform:none; }
  @keyframes chipin { from { opacity:0; transform:translateY(-160%);} to { opacity:1; transform:translateY(-120%);} }
  .rail { background:#111a2e; border:1px solid #1e293b; border-radius:12px; padding:12px; }
  .station { padding:7px 12px; border-radius:8px; margin:1px 0; border-left:4px solid #1e293b; }
  .station.active { border-left-color:#3b82f6; background:#16233d; animation: stpulse 1.2s ease-in-out infinite; }
  .station.done { border-left-color:#22c55e; }
  @keyframes stpulse { 0%,100% { background:#16233d; } 50% { background:#1b2c4d; } }
  .station-name { font-weight:bold; letter-spacing:1px; font-size:13px; }
  .station-info, .station-loop { font-size:11px; color:#64748b; }
  .station-loop { color:#f59e0b; }
  .rail-link { color:#1e293b; margin-left:18px; line-height:6px; }
  .tickets { margin-top:10px; max-height:340px; overflow-y:auto; }
  .ticket { background:#111a2e; border:1px solid #1e293b; border-radius:10px; margin:6px 0; padding:6px 10px; }
  .ticket.open { border-color:#f59e0b88; animation: tkin .3s ease-out; }
  .ticket.fixed { opacity:.75; } .ticket.stood { border-color:#64748b; }
  @keyframes tkin { from { opacity:0; transform:translateX(20px);} to { opacity:1; transform:none;} }
  .ticket summary { cursor:pointer; display:flex; gap:8px; align-items:center; list-style:none; }
  .ticket-kind { color:#f59e0b; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
  .ticket-label { font-family:monospace; font-size:12px; }
  .stamp { margin-left:auto; font-size:10px; font-weight:bold; letter-spacing:1px;
      padding:2px 8px; border-radius:4px; border:1px solid; }
  .stamp.open { color:#f59e0b; border-color:#f59e0b; }
  .stamp.fixed { color:#22c55e; border-color:#22c55e; }
  .stamp.stood { color:#94a3b8; border-color:#94a3b8; }
  .ticket-body { margin-top:8px; }
  .speaker { font-size:10px; letter-spacing:2px; color:#64748b; }
  .bubble { background:#0b1220; border:1px solid #1e293b; border-radius:8px; padding:8px;
      font-family:monospace; font-size:12px; white-space:pre-wrap; }
  .inspector { background:#111a2e; border:1px solid #1e293b; border-radius:10px; padding:10px; margin-top:10px; }
  .insp-head { display:flex; gap:10px; align-items:center; }
  .insp-id { font-family:monospace; font-weight:bold; font-size:14px; }
  .insp-state { color:white; font-size:11px; padding:2px 8px; border-radius:10px; }
  .insp-desc { color:#94a3b8; font-size:12px; margin:6px 0; }
  .insp-chain { font-size:12px; color:#cbd5e1; }
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
                  html.Div(id="inspector")]),
        html.Div([html.Div(id="rail"), html.Div(id="tickets")]),
    ], className="main"),
    dcc.Store(id="run-id"), dcc.Store(id="selected"),
    dcc.Store(id="upload-cache"),
    dcc.Interval(id="tick", interval=700),
], className="wrap")


@app.callback(Output("upload-cache", "data"),
              Input("upload", "contents"), State("upload", "filename"),
              prevent_initial_call=True)
def cache_upload(contents, filename):
    return {"contents": contents, "filename": filename}


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
              prevent_initial_call=True)
def select_entity(_clicks):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and any(_clicks):
        return trig.get("oid")
    return dash.no_update


@app.callback(
    Output("rail", "children"), Output("tickets", "children"),
    Output("instruments", "children"), Output("scene", "children"),
    Output("inspector", "children"),
    Output("scrub", "max"), Output("scrub", "value"),
    Input("tick", "n_intervals"), Input("scrub", "value"),
    State("run-id", "data"), State("selected", "data"), State("scrub", "max"))
def render(_n, scrub_value, run_id, selected, scrub_max):
    run = RUNS.get(run_id or "")
    if not run:
        empty = derive([])
        return (rail_component(empty), tickets_component(empty),
                instruments_component(empty),
                scene_component(empty, None),
                inspector_component(empty, None), 1, 1)
    events = run["events"]
    total = max(1, len(events))
    # Follow-live rule: the slider sticks to the right edge unless the user
    # dragged it left; dragging back to the edge resumes following.
    following = scrub_value is None or scrub_max is None or scrub_value >= scrub_max
    k = total if (following or ctx.triggered_id != "scrub") and following else min(scrub_value, total)
    d = derive(events[:k])
    return (rail_component(d), tickets_component(d), instruments_component(d),
            scene_component(d, run.get("image_src")),
            inspector_component(d, selected),
            total, total if following else k)


if __name__ == "__main__":
    app.run(debug=False, port=8060)
