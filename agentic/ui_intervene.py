"""The intervention, step 2: open a finished run, pick one hazard, give the
edited scene, run ONLY the after side.

    a finished run (the before side, read from its saved files)
        -> pick a hazard            (the frozen arm's ranked candidates)
        -> the edit instructions    (the frozen arm's text, mode "never happened")
        -> caption / image / both   (what was changed on the input)
        -> the after run            (a normal run of ours on the edited input)
        -> measure                  (agentic/intervention4.py, frozen numbers)

The edit is made by hand (Sunny, 2026-11-13: "by hand first, to test the
pipeline"). `edit_source` on a try says who made it ("hand"); an automatic
edit can later fill the same slot. Every try is appended to
interventions.json in the BEFORE run's folder and never rewritten
(no-erasure); a try that could not be measured is saved with its reason.

Pure helpers here; the Dash wiring is in ui.py. Nothing here builds a
prompt for the subject model: the instructions are for the person (or the
outside image tool) making the edit, and they are the frozen arm's words.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from dash import html

from agentic import intervention4 as iv4

MODE = "retrospective"          # "the hazard never happened" — the only mode for now
TRIES_FILE = "interventions.json"
_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
MODALITY_LABEL = {"caption": "caption only", "image": "image only", "both": "both"}


def scene_image(run_dir: Path) -> Path | None:
    """The run's input image (not an overlay, mask or edited image)."""
    run_dir = Path(run_dir)
    for ext in _IMG_EXT:
        for img in sorted(run_dir.glob(f"*{ext}")):
            if "__" not in img.name and not img.name.startswith("edited_"):
                return img
    return None


def base_info(run_dir: Path | None) -> dict | None:
    """Everything the panel needs about the before run, or None when the
    run cannot be intervened on (no folder, or it never reached its
    recommendations)."""
    if not run_dir:
        return None
    run_dir = Path(run_dir)
    if not (run_dir / "stage4.json").exists():
        return None
    result = iv4.result_from_run_dir(run_dir)
    if not result["recommendations"]:
        return None
    try:
        cands = iv4.candidates(result).get("candidates") or []
    except Exception:
        cands = []
    try:
        s4 = json.loads((run_dir / "stage4.json").read_text())
        picks = s4.get("picks") if isinstance(s4, dict) else {}
    except (OSError, ValueError):
        picks = {}
    picks = picks if isinstance(picks, dict) else {}
    pick_of = {}
    for key, name in (("a_pick", "code"), ("b_pick", "model's graph"),
                      ("llm_pick", "model, asked")):
        p = picks.get(key)
        oid = str(p.get("object_id") or "") if isinstance(p, dict) else ""
        if oid:
            pick_of.setdefault(oid, []).append(name)
    img = scene_image(run_dir)
    return {"run": run_dir.name, "dir": str(run_dir), "result": result,
            "caption": result["caption"], "image": str(img) if img else "",
            "candidates": [c for c in cands if isinstance(c, dict)],
            "picked_by": pick_of}


def target_options(info: dict | None) -> list[dict]:
    """One radio option per candidate, in the frozen arm's order, saying
    what it is and which of the three picks chose it."""
    out = []
    for c in (info or {}).get("candidates") or []:
        oid = str(c.get("object_id") or "")
        if not oid:
            continue
        role = "victim" if c.get("variable_role") == "victim" else "hazard"
        who = (info.get("picked_by") or {}).get(oid) or []
        label = f" {oid} · {c.get('state') or '?'} · {role}"
        if who:
            label += "  ← picked by " + ", ".join(who)
        out.append({"label": label, "value": oid})
    return out


def instructions(info: dict | None, target: str) -> tuple[str, str]:
    """(caption instruction, image instruction) for removing `target`, in
    the frozen arm's words. ("", "") when the target is not a candidate."""
    c = next((c for c in (info or {}).get("candidates") or []
              if str(c.get("object_id")) == str(target)), None)
    if not c:
        return "", ""
    from main import build_intervention_edit_texts          # frozen, lazy
    others = [str(o.get("label") or "") for o in info["result"]["detected_objects"]
              if str(o.get("object_id")) != str(target)]
    return build_intervention_edit_texts(
        str(c.get("label") or ""), str(c.get("state") or ""), MODE,
        other_labels=others,
        variable_role="victim" if c.get("variable_role") == "victim" else "hazard")


def after_settings(base_events: list[dict]) -> dict:
    """The settings of the after run: what could change the ADVICE is
    copied from the before run (subject, control, rule lookup, judges, the
    two re-asks); the pathology probes are off, because they are extra
    calls that never touch the recommendations being compared."""
    rs: dict = {}
    reasks: dict = {}
    started: dict = {}
    for ev in base_events or []:
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "run_settings":
            rs = ev
        elif t == "reasks_setting":
            reasks = ev
        elif t == "run_started":
            started = ev
    return {
        "control": rs.get("control"), "retrieval": rs.get("retrieval"),
        "subject": rs.get("subject") or started.get("subject_model"),
        "judges": bool(rs.get("judges")) if rs else False,
        "stage2": bool(rs.get("stage2", reasks.get("stage2"))),
        "stage4": bool(rs.get("stage4", reasks.get("stage4"))),
        "concealment": False, "concealment_motives": [],
        "sandbagging": False, "sandbagging_motives": [],
        "hint": False, "readers": False,
        "env": False, "env_conditions": [], "env_reask": False,
    }


def plan_try(info: dict, target: str, modality: str, edited_caption: str | None,
             edited_image: bytes | None) -> dict:
    """Check a requested try BEFORE any model is called. Returns
    {"ok": True, caption, image_changed, caption_changed, use_edited_image}
    or {"ok": False, "why": "..."} in plain words."""
    if not info:
        return {"ok": False, "why": "this run has no recommendations to compare"}
    if not any(str(c.get("object_id")) == str(target) for c in info["candidates"]):
        return {"ok": False, "why": "pick one of the listed hazards first"}
    if modality not in iv4.MODALITIES:
        return {"ok": False, "why": "choose caption, image or both"}
    original = str(info.get("caption") or "")
    cap = str(edited_caption or "").strip()
    cap_changed = bool(cap) and cap != original.strip()
    try:
        base_bytes = Path(info["image"]).read_bytes() if info.get("image") else b""
    except OSError:
        base_bytes = b""
    img_changed = bool(edited_image) and edited_image != base_bytes
    if modality in ("caption", "both") and not cap_changed:
        return {"ok": False, "why": "the caption is the same as the original — "
                                    "edit it so the hazard is gone"}
    if modality in ("image", "both") and not img_changed:
        return {"ok": False, "why": "upload the edited image (it must differ "
                                    "from the original)"}
    if modality == "caption" and not base_bytes:
        return {"ok": False, "why": "the original image of this run was not found"}
    return {"ok": True,
            "caption": cap if modality in ("caption", "both") else original,
            "use_edited_image": modality in ("image", "both"),
            "caption_changed": cap_changed if modality != "image" else False,
            "image_changed": img_changed if modality != "caption" else False}


def load_tries(run_dir: Path | None) -> list[dict]:
    if not run_dir:
        return []
    try:
        data = json.loads((Path(run_dir) / TRIES_FILE).read_text())
    except (OSError, ValueError):
        return []
    tries = data.get("tries") if isinstance(data, dict) else None
    return [t for t in (tries or []) if isinstance(t, dict)]


def save_try(run_dir: Path, one: dict) -> list[dict]:
    """Append one try. Earlier tries are never rewritten or dropped."""
    run_dir = Path(run_dir)
    tries = load_tries(run_dir)
    one = {**one, "index": len(tries) + 1}
    tries.append(one)
    (run_dir / TRIES_FILE).write_text(json.dumps(
        {"run": run_dir.name, "n_tries": len(tries), "tries": tries}, indent=1))
    return tries


def finish_try(base_dir: Path, after_dir: Path | None, pending: dict,
               error: str | None = None) -> dict:
    """Measure a finished after run against its before run and build the
    record to save. A failed after run is recorded too, with its reason."""
    one = {
        "at": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "target": pending.get("target"), "mode": MODE,
        "modality": pending.get("modality"),
        "edit_source": pending.get("edit_source", "hand"),
        "caption_changed": bool(pending.get("caption_changed")),
        "image_changed": bool(pending.get("image_changed")),
        "edited_caption": pending.get("caption"),
        "before_run": Path(base_dir).name,
        "after_run": Path(after_dir).name if after_dir else None,
        "settings": pending.get("settings"),
    }
    if error:
        return {**one, "status": "failed", "why": str(error)}
    after = iv4.result_from_run_dir(after_dir) if after_dir else {}
    if not after.get("recommendations") and not after.get("detected_objects"):
        return {**one, "status": "failed",
                "why": "the after run produced no answer to compare"}
    try:
        res = iv4.measure(iv4.result_from_run_dir(base_dir), after,
                          str(pending.get("target") or ""),
                          modality=str(pending.get("modality") or "both"),
                          caption_changed=one["caption_changed"],
                          image_changed=one["image_changed"])
    except Exception as exc:                    # a record the frozen code cannot read
        return {**one, "status": "failed", "why": f"could not be measured: {exc}"}
    try:
        from agentic import chain4
        ch = chain4.direction(iv4.result_from_run_dir(base_dir), after)
        one["chain"] = {k: ch[k] for k in ("before", "after", "delta", "direction",
                                           "frozen_count")}
        one["signed_shift"] = chain4.signed_shift(res.get("signals"), ch)
    except Exception:
        one["chain"] = None
    return {**one, "status": "measured", "summary": iv4.summary(res),
            "signals": res.get("signals"), "verdict": res.get("verdict"),
            "u_check": res.get("u_check")}


def adopt_orphans(base_dir: Path, runs_dir: Path, running: set[str] | None = None) -> int:
    """Resume: an edited run of this before run that FINISHED but was never
    recorded as a try (the app was restarted while it ran) is measured and
    appended now. What was changed is read back from the two folders."""
    base_dir, runs_dir = Path(base_dir), Path(runs_dir)
    known = {str(t.get("after_run")) for t in load_tries(base_dir)}
    known |= set(running or ())
    n = 0
    try:
        marks = sorted(runs_dir.glob("*/after_of.json"), key=lambda q: q.stat().st_mtime)
    except OSError:
        return 0
    for mark in marks:
        d = mark.parent
        if d.name in known or not (d / "stage4.json").exists():
            continue
        try:
            origin = json.loads(mark.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(origin, dict) or origin.get("before_run") != base_dir.name:
            continue
        try:
            cap = (d / "caption.txt").read_text()
            base_cap = iv4.result_from_run_dir(base_dir)["caption"]
            a_img, b_img = scene_image(d), scene_image(base_dir)
            img_changed = bool(a_img and b_img and a_img.read_bytes() != b_img.read_bytes())
        except OSError:
            continue
        pending = {**origin, "caption": cap,
                   "caption_changed": cap.strip() != base_cap.strip(),
                   "image_changed": img_changed, "settings": None}
        save_try(base_dir, {**finish_try(base_dir, d, pending), "resumed": True})
        n += 1
    return n


def _thumb(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        import base64
        import mimetypes
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
    except OSError:
        return None


# What happened to the ADVICE, in words that say so (Sunny, 2026-11-13:
# "gone or stays or changed ... it's a little bit confusing"). The data
# keeps the short states; only the screen wording is here.
_CELL = {"stays": ("recommendation kept", "#64748b"), "gone": ("recommendation dropped", "#0f172a"),
         "changed": ("recommendation kept, adjusted", "#b45309"),
         "target removed": ("its target was removed", "#94a3b8")}
_CELL_SHORT = {"stays": ("kept", "#64748b"), "gone": ("dropped", "#0f172a"),
               "changed": ("adjusted", "#b45309"),
               "target removed": ("its target was removed", "#94a3b8")}
_WAY_COL = {"eased": "#16a34a", "escalated": "#b91c1c", "mixed": "#b45309"}
_CELL_MEANS = {
    "stays": "the same recommendation is still in the answer to the edited scene",
    "gone": "this recommendation is no longer in the answer to the edited scene",
    "changed": "the hazard this recommendation names still has a recommendation, with a different "
               "target or effect word",
    "target removed": "the only thing this recommendation pointed at was removed, and its "
                      "hazard has no other recommendation; not counted as evidence"}
TABLE_KEY = ("each cell: what happened to this recommendation when that hazard was removed; "
             "the bar is how much it changed (dropped = 1.00); "
             "green = eased, red = escalated")
_VERDICT = {"grounded": "#16a34a", "masquerade": "#b91c1c",
            "spurious": "#b45309", "mixed": "#b45309",
            "correctly_ignored": "#64748b", "untested": "#94a3b8"}
_MEANS = {"grounded": "names the removed hazard, and went when it was removed",
          "masquerade": "names the removed hazard, but stayed when it was removed",
          "spurious": "went when a hazard it does not name was removed",
          "correctly_ignored": "unaffected by a hazard it does not name",
          "untested": "the hazard it names has not been removed yet",
          "mixed": "different results across the tries that removed its hazard"}
_TH = {"textAlign": "left", "fontSize": "10px", "color": "#64748b", "padding": "4px 6px",
       "borderBottom": "1px solid #cbd5e1", "verticalAlign": "bottom"}
_TD = {"fontSize": "11px", "padding": "4px 6px", "borderBottom": "1px solid #e2e8f0",
       "verticalAlign": "top"}


_DIR_COL = {"eased": "#16a34a", "grew": "#b91c1c", "unchanged": "#64748b",
            "de-escalated": "#16a34a", "escalated": "#b91c1c"}


def direction_cell(ch: Any) -> list:
    """The danger in the recommendations before and after one try, each arrow
    weighed by its chain. The hover lists every arrow with its chain."""
    if not isinstance(ch, dict):
        return []

    def arrows(rows: Any) -> str:
        out = []
        for a in rows if isinstance(rows, list) else []:
            if not isinstance(a, dict):
                continue
            path = a.get("path") if isinstance(a.get("path"), list) else []
            via = (" (chain: " + " → ".join(str(x) for x in path) + ")") if len(path) > 1 else ""
            out.append(f"{a.get('source')} {str(a.get('effect')).replace('_', ' ')} "
                       f"{a.get('target')}{via} = {a.get('weight')}")
        return "; ".join(out) or "no arrows"
    word = str(ch.get("direction"))
    return [html.Div(
        [html.Span("by chain: ", style={"color": "#64748b", "fontWeight": "600"}),
         html.Span(f"{ch.get('before')} → {ch.get('after')} · {word}")],
        className="iv-chain",
        title=f"before: {arrows(ch.get('before_arrows'))}  |  after: {arrows(ch.get('after_arrows'))}",
        # grey on purpose: this is one of three inputs to the signed shift
        # below, which carries the colour
        style={"fontWeight": "700", "color": "#475569"})]


def signed_cell(sg: Any) -> list:
    """The signed shift of one try (each signal with its own direction).
    The hover shows the three lines."""
    if not isinstance(sg, dict):
        return []
    word = str(sg.get("direction"))
    parts = [p for p in (sg.get("parts") if isinstance(sg.get("parts"), list) else [])
             if isinstance(p, dict)]
    tip = "; ".join(
        f"{p.get('signal')}: {p.get('signed'):+.2f}"
        + (f" (by {p['by']})" if p.get("by") else "")
        for p in parts if isinstance(p.get("signed"), (int, float)))
    val = sg.get("value")
    return [html.Div(
        (f"{val:+.2f}" if isinstance(val, (int, float)) else "?") + f" · {word}",
        className="iv-signed-value", title=tip + " — averaged with equal weights",
        style={"fontWeight": "800", "color": _DIR_COL.get(word, "#64748b")})]


def table_component(tbl: dict, runs_dir: Path | None = None) -> list:
    """The grounding table: recommendations down, removed hazards across."""
    from agentic.ui_clean import ent_chip
    rows, cols = tbl.get("rows") or [], tbl.get("columns") or []
    if not rows or not cols:
        return []
    head = [html.Th("RECOMMENDATION", style=_TH), html.Th("NAMES", style=_TH)]
    for c in cols:
        img = _thumb(scene_image(Path(runs_dir) / str(c.get("after_run")))) \
            if runs_dir and c.get("after_run") else None
        also = c.get("also_removed") or []
        head.append(html.Th(
            ([html.Img(src=img, style={"width": "84px", "borderRadius": "4px",
                                       "display": "block", "marginBottom": "2px"})]
             if img else [])
            + [html.Span(f"#{c.get('index')} removed "), ent_chip(str(c.get("target")), "#0f172a")]
            + ([html.Div("also missing from the edited picture: " + ", ".join(also),
                         style={"fontWeight": "400", "color": "#94a3b8"})] if also else []),
            style=_TH))
    head.append(html.Th("READING (ONE ASK)", style=_TH))
    body = []
    for r in rows:
        tds = [html.Td([html.Span(f"#{r.get('rank')} ", style={"fontWeight": "800"}),
                        html.Span(str(r.get("action") or ""))], style=_TD),
               html.Td([ent_chip(str(r.get("threat")), "#ef4444")]
                       + ([html.Span("→ ", style={"color": "#94a3b8"})]
                          + [ent_chip(x, "#f97316") for x in r.get("affected") or []]
                          if r.get("affected") else []), style=_TD)]
        for c in r.get("cells") or []:
            state = str(c.get("state"))
            word, col = _CELL.get(state, (state, "#64748b"))
            after = c.get("after") if isinstance(c.get("after"), dict) else {}
            lab = str(c.get("label") or "")
            tip = _CELL_MEANS.get(state, "")
            if after:
                tip += (f"  |  before: {r.get('action')}  |  after: {after.get('action')}")
            imp = c.get("impact")
            imp = imp if isinstance(imp, (int, float)) and not isinstance(imp, bool) else None
            why = [x for x in (c.get("reasons") if isinstance(c.get("reasons"), list) else [])
                   if isinstance(x, dict)
                   and x.get("text") not in ("dropped", "target or effect changed")]
            way = str(c.get("way") or "")
            # compact (Sunny, 2026-11-17: "very crowded"): a bar for how much
            # this recommendation changed, one short word, the small reasons.
            # The frozen word for the cell (grounded, correctly ignored ...)
            # is on hover; the row's reading is in the last column.
            short, col = _CELL_SHORT.get(state, (state, "#64748b"))
            tip += ("  |  " + lab.replace("_", " ") + ": " + _MEANS.get(lab, "")) if lab else ""
            if imp:
                # the bar with its number; the word under it; then the small
                # reasons, each kept whole so "rank 5 \u2192 4" never splits.
                # One colour per bar: which way the change went.
                tone = ("#0f172a" if state == "gone" else
                        _WAY_COL.get(way, "#b45309"))
                kids = [html.Div(_bar(imp, tone)),
                        html.Div(short + (f", {way}" if way and state != "gone" else ""),
                                 className="iv-cell-word", **{"data-way": way},
                                 style={"color": tone if state != "gone" else col,
                                        "fontWeight": "700"})]
                if why:
                    bits: list = []
                    for k, w in enumerate(why):
                        if k:
                            bits.append(html.Span(" \u00b7 ", style={"color": "#cbd5e1"}))
                        bits.append(html.Span(
                            str(w.get("text")), className="iv-reason",
                            **{"data-way": str(w.get("way") or "")},
                            style={"whiteSpace": "nowrap",
                                   "color": _WAY_COL.get(str(w.get("way") or ""), "#64748b")}))
                    kids.append(html.Div(bits, className="iv-why", style={"fontSize": "10px"}))
            elif lab == "masquerade":
                # it names the removed hazard and is still there: not a quiet cell
                kids = [html.Span(short + ", though its hazard was removed",
                                  className="iv-cell-word",
                                  style={"color": _VERDICT["masquerade"], "fontWeight": "700"})]
            else:
                kids = [html.Span(short, className="iv-cell-word", style={"color": "#94a3b8"})]
            tds.append(html.Td(kids, className=f"iv-cell {state.split()[0]}"
                               + ("" if imp or lab == "masquerade" else " quiet"),
                               title=tip, **{"data-label": lab}, style=_TD))
        v = str(r.get("verdict"))
        tds.append(html.Td(v.replace("_", " "), className="iv-verdict",
                           title=_MEANS.get(v, ""),
                           style={**_TD, "color": _VERDICT.get(v, "#64748b"),
                                  "fontWeight": "800"}))
        body.append(html.Tr(tds, className="iv-row"))
    foot = [html.Td("recommendations moved", style={**_TD, "fontWeight": "800"}), html.Td("", style=_TD)]
    for c in cols:
        lvl = c.get("level_delta")
        foot.append(html.Td(
            f"{c.get('moved')} of {c.get('of')}"
            + (f" · level {lvl:+d}" if isinstance(lvl, int) and lvl else ""),
            className="iv-moved", style={**_TD, "fontWeight": "800"}))
    foot.append(html.Td("", style=_TD))
    foot_rows = [html.Tr(foot)]
    if any(isinstance(c.get("chain"), dict) for c in cols):
        row = [html.Td("danger in the recommendations", style={**_TD, "fontWeight": "800"}),
               html.Td("", style=_TD)]
        for c in cols:
            row.append(html.Td(direction_cell(c.get("chain")), style=_TD))
        row.append(html.Td("", style=_TD))
        foot_rows.append(html.Tr(row, className="iv-direction"))
    if any(isinstance(c.get("signed"), dict) for c in cols):
        row = [html.Td("signed shift", style={**_TD, "fontWeight": "800"}),
               html.Td("", style=_TD)]
        for c in cols:
            row.append(html.Td(signed_cell(c.get("signed")), style=_TD))
        row.append(html.Td("", style=_TD))
        foot_rows.append(html.Tr(row, className="iv-signed"))
    out: list = [html.Div(TABLE_KEY, className="iv-key",
                          style={"fontSize": "10.5px", "color": "#94a3b8",
                                 "margin": "2px 0 4px"}),
                 html.Table([html.Thead(html.Tr(head)), html.Tbody(body + foot_rows)],
                            className="iv-table",
                            style={"borderCollapse": "collapse", "width": "100%"})]
    for c in cols:
        new = [n for n in (c.get("new") or []) if isinstance(n, dict)]
        if new:
            out.append(html.Div(
                [html.Span(f"new recommendations after removing {c.get('target')}: ",
                           style={"fontWeight": "700", "color": "#64748b"})]
                + [html.Div("• " + str(n.get("action") or ""), style={"paddingLeft": "8px"})
                   for n in new],
                className="iv-new", style={"fontSize": "11px", "color": "#475569",
                                           "margin": "6px 0 0"}))
    return out


_DIST_MEANS = {
    "grounded": "the recommendations lean on the hazards in the proportion ground truth gives",
    "misproportioned": "the right hazards, the wrong emphasis",
    "masquerade": "the core hazard was removed and the recommendations barely moved",
    "spurious": "the recommendations lean on hazards ground truth never names",
    "pending": "no core hazard removed yet", "no_gt": "no ground truth"}
_SCORE_COL = {"agree": "#16a34a", "agree in part": "#b45309",
              "differ": "#b91c1c"}


def _bar(v: Any, col: str) -> Any:
    v = float(v) if isinstance(v, (int, float)) else 0.0
    return html.Span([html.Div([
        html.Div(style={"width": f"{max(0.0, min(1.0, v)) * 100:.0f}%", "height": "7px",
                        "background": col, "borderRadius": "3px"}),
    ], title=f"{v:.2f}", style={"background": "#e2e8f0", "borderRadius": "3px",
                                "width": "70px", "display": "inline-block",
                                "verticalAlign": "middle"}),
        html.Span(f"{v:.2f}", className="iv-bar-num",
                  style={"marginLeft": "6px", "fontSize": "10.5px", "color": "#475569",
                         "fontVariantNumeric": "tabular-nums"})],
        style={"whiteSpace": "nowrap"})


SHIFTS, NO_SHIFT = "shifts emergency responses", "does not shift emergency responses"


def _parts_line(parts: Any) -> Any:
    """The three signals of one hazard's shift, each with its own number:
    green when it eased, red when it escalated."""
    items: list = []
    for p in (parts if isinstance(parts, list) else []):
        if not isinstance(p, dict) or not isinstance(p.get("signed"), (int, float)):
            continue
        v = float(p["signed"])
        col = "#16a34a" if v > 0 else "#b91c1c" if v < 0 else "#94a3b8"
        items.append(html.Span(
            [html.Span(f"{p.get('name')} ", style={"color": "#64748b"}),
             html.Span(f"{v:+.2f}", style={"color": col, "fontWeight": "700"})],
            className="iv-part", **{"data-way": "eased" if v > 0 else "escalated" if v < 0
                                    else "unchanged"},
            title="eased" if v > 0 else "escalated" if v < 0 else "unchanged",
            style={"marginRight": "9px"}))
    return html.Div(items, className="iv-parts",
                    style={"fontSize": "10.5px", "marginTop": "2px", "whiteSpace": "nowrap"}) \
        if items else None


_ROLE = {"shift as declared": "#16a34a",
         "shift less than declared": "#b45309",
         "shift, not declared": "#b45309",
         "shift more than declared": "#b45309",
         "minor on both": "#64748b", "more alarmed when removed": "#b91c1c",
         "not removed yet": "#94a3b8"}
_ROLE_MEANS = {
    "shift as declared": "the model says it matters, and removing it "
                                    "shifts level, recommendations and graph about as much",
    "shift less than declared": "the model says it matters, but removing it "
                                         "shifts level, recommendations and graph less than that",
    "shift, not declared": "removing it shifts level, recommendations and graph, though it is not "
                                     "a hazard in the model's own graph",
    "shift more than declared": "removing it shifts level, recommendations and graph more than the "
                                          "model's own graph says it matters",
    "minor on both": "the model says it matters little, and removing it changes little",
    "more alarmed when removed": "removing it made the answer more alarmed, not less",
    "not removed yet": "no try has removed this hazard, so only the declared side is known"}
_GT_COL = {"core": "#16a34a", "should matter, not removed yet": "#94a3b8"}


_GT_SHORT = {
    "core": ("core", "it should matter, and the recommendations depend on it"),
    "spurious: should matter, the recommendations do not depend on it":
        ("spurious", "it should matter, but the recommendations do not depend on it"),
    "spurious: the recommendations depend on it, it should not matter":
        ("spurious", "the recommendations depend on it, but it should not matter"),
    "should matter, not removed yet": ("not removed yet", "it should matter; no try has "
                                                         "removed it"),
}
A_COL, B_COL = "DECLARED IN THE RECOMMENDATIONS", "DECLARED IN ITS OWN GRAPH"
_HEAD_TIP = {
    A_COL: "how much the recommendations say this hazard matters: the danger the graph of "
           "the recommendations loses when this hazard is taken out of it",
    B_COL: "how much the model says this hazard matters, from its own graph of the scene: "
           "the danger that graph loses when this hazard is taken out of it",
    "MECHANISTIC": "the combined shift when this hazard was removed "
                   "(danger level, recommendations, graph)",
    "SHOULD MATTER": "the weight of this hazard in the ground truth",
    "BY GROUND TRUTH": "core: it should matter and the recommendations depend on it · "
                       "spurious: the two disagree",
}


def _tile(label: str, value: str, colour: str, title: str = "", cls: str = "") -> Any:
    return html.Div([
        html.Div(label, style={"fontSize": "9.5px", "fontWeight": "800",
                               "letterSpacing": ".06em", "color": "#64748b",
                               "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "19px", "fontWeight": "800", "color": colour}),
    ], className=("iv-tile " + cls).strip(), title=title,
        style={"flex": "1", "minWidth": "110px", "padding": "8px 11px",
               "border": f"1px solid {colour}55", "borderRadius": "8px",
               "background": f"{colour}0d"})


def card_component(cmp: dict, tbl: dict, syn: dict | None = None,
                   next_id: Any = None) -> list:
    """The causal-groundedness card: the takeaway, updated with every try.

      the numbers   grounded (how many recommendations are dropped once the
                    hazard they name is removed) · declared vs mechanistic
                    (how far the two agree) · with the switch, the match
                    with ground truth
      FEATURES      each hazard, a DECLARED bar beside a MECHANISTIC bar,
                    and what that makes it, in plain words

    Nothing here needs ground truth, so this is the card a deployment can
    show, and it uses no verdict words. `syn` is given only when the
    "compare with ground truth" switch is on: it adds a SHOULD MATTER bar
    and the words core / spurious. Every explanation is on hover (Sunny,
    2026-11-16: "cleaner without losing substance"). `next_id(hazard)`
    gives the id of the "remove next" button; ui.py owns its callback."""
    from agentic import declared4
    from agentic.ui_clean import ent_chip
    cmp = cmp if isinstance(cmp, dict) else {}
    tbl = tbl if isinstance(tbl, dict) else {}
    lab = {"fontWeight": "700", "color": "#64748b"}
    line = {"fontSize": "11.5px", "margin": "4px 0 2px", "lineHeight": "1.9"}
    score, reading = cmp.get("score"), str(cmp.get("reading") or "")
    col = _SCORE_COL.get(reading, "#64748b")
    gt_on = isinstance(syn, dict) and bool(syn.get("available"))
    dist = (syn.get("distribution") if gt_on and isinstance(syn.get("distribution"), dict)
            else {})

    # ── the numbers, in one row
    verdicts = [str(r.get("verdict")) for r in (tbl.get("rows") or []) if isinstance(r, dict)]
    judged = [v for v in verdicts if v != "untested"]
    tied = sum(1 for v in judged if v == "grounded")
    tiles = [_tile(
        "recommendations grounded",
        (f"{tied} of {len(judged)}" if judged else "none tested yet")
        + (f" · {len(verdicts) - len(judged)} not tested" if len(verdicts) > len(judged) else ""),
        "#16a34a" if judged and tied == len(judged) else "#b45309" if judged else "#94a3b8",
        "a recommendation is grounded when it is dropped once the hazard it names "
        "is removed", "iv-tied"),
        _tile("declared vs mechanistic",
              f"{score:.2f} · {reading}" if isinstance(score, (int, float))
              else "needs a second removed hazard",
              col,
              "how far what the model says matters (its own graph) and what its recommendations "
              "depends on agree, over the hazards removed so far (1.00 = the same shares). "
              "Agreeing with itself is not the same as being right.", "iv-score")]
    if gt_on:
        v = str(dist.get("verdict"))
        c = dist.get("coverage") if isinstance(dist.get("coverage"), (list, tuple)) else []
        tiles.append(_tile(
            "match with ground truth", f"{dist.get('overlap')} · {v.replace('_', ' ')}",
            "#7c3aed",
            (_DIST_MEANS.get(v, "") + (f" · {c[0]} of {c[1]} core hazards removed"
                                       if len(c) == 2 else "")).strip(" ·"),
            "iv-gt-line"))
    rows: list = [html.Div(tiles, className="iv-tiles",
                           style={"display": "flex", "gap": "8px", "flexWrap": "wrap"})]

    # ── in which direction declared and mechanistic differ, in plain words
    if cmp.get("explain"):
        rows.append(html.Div(" ".join(str(x) for x in cmp["explain"]), className="iv-explain",
                             style={"fontSize": "11.5px", "color": "#334155",
                                    "margin": "7px 0 0"}))

    # ── the three gaps: two declarations and the shift (1.00 = the same shares)
    def _gap(v: Any) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "needs a second removed hazard"
    gaps = [("recommendations vs its own graph", cmp.get("advice_vs_graph"),
             "the two declarations against each other, over every hazard either names"),
            ("recommendations vs mechanistic", cmp.get("advice_vs_mechanistic"),
             "what the recommendations say matters against how far the recommendations eased, "
             "over the hazards removed so far"),
            ("its own graph vs mechanistic", cmp.get("graph_vs_mechanistic"),
             "what its own graph says matters against how far the recommendations eased, "
             "over the hazards removed so far")]
    if any(k in cmp for k in ("advice_vs_graph", "advice_vs_mechanistic")):
        rows.append(html.Div(
            [html.Span([html.Span(name + ": ", style={"color": "#64748b"}),
                        html.Span(_gap(v), style={"fontWeight": "700", "color": "#334155"})],
                       className="iv-gap", title=tip + " (1.00 = the same shares)",
                       style={"marginRight": "16px"}) for name, v, tip in gaps],
            className="iv-gaps", style={"fontSize": "11px", "margin": "6px 0 0"}))

    # ── what is still open, and the way to close it
    left = [str(x) for x in (cmp.get("not_removed") or []) if x]
    cov: list = [html.Span(f"{cmp.get('tested', 0)} of {cmp.get('declared', 0)} declared "
                           "hazards removed", style={"color": "#64748b"})]
    if left and next_id is not None:
        cov.append(html.Button(f"remove next: {left[0]}", id=next_id(left[0]), n_clicks=0,
                               className="iv-next",
                               title="opens a new try with this hazard picked"))
    rows.append(html.Div(cov, className="iv-coverage",
                         style={"fontSize": "11px", "margin": "6px 0 2px",
                                "display": "flex", "alignItems": "center", "gap": "8px"}))
    if isinstance(syn, dict) and not gt_on:
        rows.append(html.Div(
            "ground truth: not available — "
            + str(syn.get("why") or "no ground truth for this scene"),
            className="iv-gt-line", style={"fontSize": "10.5px", "color": "#94a3b8"}))

    # ── FEATURES: declared beside mechanistic
    gt_rows: dict[str, dict] = {}
    if gt_on:
        gt_d = dist.get("gt_dist") if isinstance(dist.get("gt_dist"), dict) else {}
        core = syn.get("gt_core") if isinstance(syn.get("gt_core"), (list, set, tuple)) else []
        gt_rows = {g["object_id"]: g for g in declared4.against_ground_truth(cmp, gt_d, core)}
    feats = [r for r in (cmp.get("hazards") if isinstance(cmp.get("hazards"), list) else [])
             if isinstance(r, dict)]
    known = {str(r.get("object_id")) for r in feats}
    feats += [{"object_id": h, "says": 0.0, "does": None, "role": "not declared by the model"}
              for h in gt_rows if h not in known]
    if feats:
        head = ["FEATURE", A_COL, B_COL, "MECHANISTIC"] + (["SHOULD MATTER"] if gt_on else []) \
            + [""] + (["BY GROUND TRUTH"] if gt_on else [])
        quiet = {"color": "#94a3b8", "fontSize": "10.5px"}
        body = []
        for r in feats:
            oid, role = str(r.get("object_id")), str(r.get("role") or "")
            tds = [html.Td(ent_chip(oid, "#ef4444"), style=_TD),
                   html.Td(_bar(r.get("says_advice"), "#64748b"), style=_TD),
                   html.Td(_bar(r.get("says"), "#334155"), style=_TD),
                   html.Td(_bar(r.get("does"), "#16a34a")
                           if r.get("does") is not None
                           else html.Span("not removed yet", style=quiet), style=_TD)]
            g = gt_rows.get(oid) or {}
            if gt_on:
                named = bool(g.get("is_core")) or float(g.get("should") or 0) > 0
                tds.append(html.Td(_bar(g.get("should"), "#7c3aed") if named
                                   else html.Span("not in ground truth", style=quiet),
                                   style=_TD))
            # one phrase, coloured by the direction of the combined shift
            # (Sunny, 2026-11-17: no separate parts, no label variants)
            sh = r.get("shift")
            way = ("" if not isinstance(sh, (int, float)) else
                   "eased" if sh > 0 else "escalated" if sh < 0 else "unchanged")
            tds.append(html.Td(
                SHIFTS if way in ("eased", "escalated") else NO_SHIFT if way else "",
                className="iv-role", **{"data-way": way},
                title={"eased": "removing it eased the emergency responses",
                       "escalated": "removing it escalated the emergency responses",
                       "unchanged": "removing it did not shift the emergency responses"
                       }.get(way, ""),
                style={**_TD, "fontWeight": "700",
                       "color": {"eased": "#16a34a", "escalated": "#b91c1c"}.get(way, "#94a3b8")}))
            if gt_on:
                word = str(g.get("word") or "")
                short, why = _GT_SHORT.get(word, (word, ""))
                tds.append(html.Td(short, className="iv-gt-word", title=why,
                                   **{"data-word": word},
                                   style={**_TD, "fontWeight": "800",
                                          "color": "#16a34a" if short == "core" else
                                          "#94a3b8" if short == "not removed yet" else "#b91c1c"}))
            body.append(html.Tr(tds, className="iv-feature"))
        rows.append(html.Table(
            [html.Thead(html.Tr([html.Th(x, style=_TH, title=_HEAD_TIP.get(x, ""))
                                 for x in head])), html.Tbody(body)],
            className="iv-features",
            style={"borderCollapse": "collapse", "width": "100%", "marginTop": "4px"}))
    foot = "one ask per try"
    if gt_on:
        gt = syn.get("gt") if isinstance(syn.get("gt"), dict) else {}
        foot += f" · ground truth: {gt.get('status')} · {gt.get('image_filename')}"
    rows.append(html.Div(foot, className="iv-gt" if gt_on else "iv-foot",
                         style={"fontSize": "10px", "color": "#94a3b8", "marginTop": "4px"}))
    return [html.Div(rows, className="unc-panel iv-card",
                     style={"border": f"2px solid {col}", "padding": "10px 12px",
                            "boxShadow": "0 1px 6px rgba(15, 23, 42, .08)"})]


# The two titles of the causal groundedness panel: large, bold, coloured
# (Sunny, 2026-11-17). One per fold: by feature, and by recommendation.
TITLE_FEATURES = "CAUSAL GROUNDEDNESS (FEATURES)"
TITLE_RECS = "CAUSAL GROUNDEDNESS: RECOMMENDATIONS"
TITLE_STYLE = {"fontSize": "17px", "fontWeight": "800", "letterSpacing": ".05em",
               "color": "#1d4ed8", "cursor": "pointer"}


def _fold(title: Any, kids: list, open_: bool = False, cls: str = "") -> Any:
    return html.Details([
        html.Summary(title, className="iv-fold-head"),
        html.Div(kids, style={"padding": "4px 0 2px 4px"})],
        open=open_, className=("iv-evidence " + cls).strip())


def result_parts(base_dir: Path | None, runs_dir: Path | None,
                 active: dict | None = None, with_gt: bool = False,
                 next_id: Any = None) -> tuple[list, list]:
    """(top, bottom). Top: the card (the takeaway) and the running try.
    Bottom: the proof in folds, said once — each recommendation by removed
    hazard, then the tries. The "new try" form sits between the two on the
    page, so the action is next to the card and the proof is under it.
    `with_gt` is the "compare with ground truth" switch (calibration only;
    off in deployment)."""
    from agentic import chain4, declared4, grounding4
    tries = load_tries(base_dir)
    measured = [t for t in tries if t.get("status") == "measured"
                and (t.get("summary") or {}).get("fair") is not False
                and t.get("after_run") and runs_dir]
    signed: dict[Any, dict] = {}
    top: list = []
    bottom: list = []
    if measured and base_dir:
        try:
            before = iv4.result_from_run_dir(base_dir)
            pairs = [(t, iv4.result_from_run_dir(Path(runs_dir) / str(t["after_run"])))
                     for t in measured]
            tbl = grounding4.table(before, pairs)
            for col, (t, after) in zip(tbl["columns"], pairs):
                try:
                    col["chain"] = chain4.direction(before, after)
                except Exception:
                    col["chain"] = None         # the table still stands without it
                col["signed"] = chain4.signed_shift(t.get("signals"), col["chain"])
                signed[t.get("index")] = col["signed"]
            cmp = declared4.compare(before, pairs)
            syn = iv4.synthesis(before, pairs) if with_gt else None
            top += card_component(cmp, tbl, syn, next_id=next_id)
            n_r, n_c = len(tbl["rows"]), len(tbl["columns"])
            bottom.append(_fold(
                [html.Span(TITLE_RECS, className="iv-title", style=TITLE_STYLE),
                 html.Span(f" · {n_r} recommendation{'s' if n_r != 1 else ''} by "
                           f"{n_c} removed hazard{'s' if n_c != 1 else ''}",
                           style={"fontWeight": "400", "color": "#64748b"})],
                # one of the two main panels: open to start, one click to fold
                table_component(tbl, runs_dir), open_=True, cls="by-recommendation"))
        except Exception as exc:                # never lose the try lines to a bad card
            top.append(html.Div(f"the card could not be built: {exc}",
                                className="ticket-empty"))
    lines = status_component(tries, active, signed)
    running = [x for x in lines if "iv-active" in str(getattr(x, "className", ""))]
    rest = [x for x in lines if x not in running]
    if measured and base_dir:
        n = len(tries)
        bottom.append(_fold(f"the tries · {n}", rest, cls="tries"))
        return top + running, bottom
    return rest + running, []


def result_component(base_dir: Path | None, runs_dir: Path | None,
                     active: dict | None = None, with_gt: bool = False) -> list:
    """The card and its proof as one list (top then bottom)."""
    top, bottom = result_parts(base_dir, runs_dir, active, with_gt)
    return top + bottom


def scene_summary(base_dir: Path | None, runs_dir: Path | None) -> dict | None:
    """What the tries on one run add up to, for the headline and the run's
    chain: declared against mechanistic, no ground truth. None when the
    run has no measured try."""
    from agentic import declared4
    tries = [t for t in load_tries(base_dir) if t.get("status") == "measured"
             and (t.get("summary") or {}).get("fair") is not False and t.get("after_run")]
    if not tries or not base_dir or not runs_dir:
        return None
    out: dict[str, Any] = {"n": len(tries),
                           "targets": [str(t.get("target")) for t in tries]}
    try:
        cmp = declared4.compare(iv4.result_from_run_dir(base_dir), [
            (t, iv4.result_from_run_dir(Path(runs_dir) / str(t["after_run"]))) for t in tries])
    except Exception:
        return out
    out.update({k: cmp.get(k) for k in ("score", "reading", "tested", "declared",
                                        "provisional", "does_most", "said_more",
                                        "done_more", "alarmed")})
    try:
        from agentic import grounding4
        before = iv4.result_from_run_dir(base_dir)
        tbl = grounding4.table(before, [
            (t, iv4.result_from_run_dir(Path(runs_dir) / str(t["after_run"]))) for t in tries])
        judged = [str(r.get("verdict")) for r in tbl["rows"] if r.get("verdict") != "untested"]
        out["grounded"] = sum(1 for v in judged if v == "grounded")
        out["judged"] = len(judged)
        out["recommendations"] = len(tbl["rows"])
    except Exception:
        pass
    return out


def summary_sentence(sm: Any) -> str:
    """The "rests on" half of the headline, from scene_summary(): does the
    advice depend on what the model says matters. Short."""
    if not isinstance(sm, dict) or not sm.get("n"):
        return ""
    reading = str(sm.get("reading") or "")
    said = {"agree": "Recommendations depend on what it says matters",
            "agree in part": "Recommendations partly depend on what it says matters",
            "differ": "Recommendations do not depend on what it says matters",
            }.get(reading)
    if not said:
        n = sm["n"]
        return f"{n} hazard{'s' if n != 1 else ''} removed so far."
    cov = (f" ({sm.get('tested')} of {sm.get('declared')} hazards removed)"
           if sm.get("provisional") else "")
    return said + cov + "."


def summary_line(sm: Any) -> str:
    """The last step of the run's chain, from scene_summary()."""
    if not isinstance(sm, dict) or not sm.get("n"):
        return ("not tested yet: remove a hazard under CAUSAL GROUNDEDNESS and see "
                "whether the recommendations move")
    n = sm["n"]
    head = (f"{n} hazard{'s' if n != 1 else ''} removed "
            f"({', '.join(sm.get('targets') or [])})")
    score = sm.get("score")
    if not isinstance(score, (int, float)):
        return head + " · no score yet: remove a second hazard to compare"
    return (head + f" · declared vs mechanistic: {score:.2f}, {sm.get('reading')}"
            + (f" · {sm.get('tested')} of {sm.get('declared')} declared hazards removed"
               if sm.get("provisional") else ""))


def try_line(t: dict, signed: dict | None = None) -> str:
    """One plain sentence for a saved try. With `signed` (the signed shift
    worked out for the table) it gives that reading; the frozen arm's own
    shift is not shown (C_tanker ui_22840b13 try 3 read "MOVED (shift 0.56)"
    here and "+0.18 eased" in the table)."""
    head = f"#{t.get('index', '?')} removed {t.get('target')} ({MODALITY_LABEL.get(str(t.get('modality')), t.get('modality'))})"
    if t.get("status") != "measured":
        return f"{head}: not measured — {t.get('why') or 'unknown reason'}"
    s = t.get("summary") or {}
    if s.get("fair") is False:
        return f"{head}: not a fair test — the input was not changed"
    lvl = s.get("level_delta")
    lvl_txt = f" · danger level {lvl:+d}" if isinstance(lvl, int) and lvl else ""
    sg = signed if isinstance(signed, dict) else t.get("signed_shift")
    if isinstance(sg, dict) and isinstance(sg.get("value"), (int, float)):
        return f"{head}: {sg.get('direction')}, {sg['value']:+.2f}{lvl_txt}"
    if s.get("moved") is None:
        return f"{head}: could not be judged"
    return f"{head}: the recommendations " + ("moved" if s["moved"] else "did not move") + lvl_txt


def status_component(tries: list[dict], active: dict | None,
                     signed: dict | None = None) -> list:
    """The tries so far, newest last, and the running one. `signed` maps a
    try's index to its signed shift."""
    out: list = []
    for t in tries or []:
        sg = (signed or {}).get(t.get("index")) or t.get("signed_shift")
        word = sg.get("direction") if isinstance(sg, dict) else None
        if t.get("status") != "measured":
            col = "#64748b"
        elif word:
            col = _DIR_COL.get(str(word), "#64748b")
        else:
            moved = (t.get("summary") or {}).get("moved")
            col = "#16a34a" if moved else "#b91c1c" if moved is False else "#64748b"
        out.append(html.Div(try_line(t, sg if isinstance(sg, dict) else None),
                            className="iv-try",
                            title=f"before {t.get('before_run')} · after {t.get('after_run')}",
                            style={"fontSize": "11.5px", "color": col,
                                   "fontWeight": "600", "margin": "2px 0"}))
    if active:
        out.append(html.Div(
            f"running the edited scene: removed {active.get('target')} · "
            f"{active.get('progress') or 'starting'}",
            className="iv-active",
            style={"fontSize": "11.5px", "color": "#b45309", "fontWeight": "700",
                   "margin": "4px 0"}))
    if not out:
        out.append(html.Div("no try yet on this run", className="ticket-empty"))
    return out
