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
    calls that never touch the advice being compared."""
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
_CELL = {"stays": ("advice kept", "#64748b"), "gone": ("advice dropped", "#0f172a"),
         "changed": ("advice kept, adjusted", "#b45309"),
         "target removed": ("nothing left to protect", "#94a3b8")}
_CELL_MEANS = {
    "stays": "the same advice is still in the answer to the edited scene",
    "gone": "this advice is no longer in the answer to the edited scene",
    "changed": "the advice is still there, but it now protects different things "
               "or uses a different effect word",
    "target removed": "what this advice protected was removed, so it had to go; "
                      "not counted as evidence"}
TABLE_KEY = "each cell: what happened to this advice when that hazard was removed"
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
    """The danger in the advice before and after one try, each arrow
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
            # one line: what happened to the advice -> the label that follows
            tds.append(html.Td(
                [html.Span(word)]
                + ([html.Span(" → ", style={"color": "#94a3b8", "fontWeight": "400"}),
                    html.Span(lab.replace("_", " "), className="iv-cell-label",
                              title=_MEANS.get(lab, ""),
                              style={"color": _VERDICT.get(lab, "#94a3b8")})] if lab else []),
                className=f"iv-cell {state.split()[0]}", title=tip,
                style={**_TD, "color": col, "fontWeight": "700"}))
        v = str(r.get("verdict"))
        tds.append(html.Td(v.replace("_", " "), className="iv-verdict",
                           title=_MEANS.get(v, ""),
                           style={**_TD, "color": _VERDICT.get(v, "#64748b"),
                                  "fontWeight": "800"}))
        body.append(html.Tr(tds, className="iv-row"))
    foot = [html.Td("advice moved", style={**_TD, "fontWeight": "800"}), html.Td("", style=_TD)]
    for c in cols:
        lvl = c.get("level_delta")
        foot.append(html.Td(
            f"{c.get('moved')} of {c.get('of')}"
            + (f" · level {lvl:+d}" if isinstance(lvl, int) and lvl else ""),
            className="iv-moved", style={**_TD, "fontWeight": "800"}))
    foot.append(html.Td("", style=_TD))
    foot_rows = [html.Tr(foot)]
    if any(isinstance(c.get("chain"), dict) for c in cols):
        row = [html.Td("danger in the advice", style={**_TD, "fontWeight": "800"}),
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
                [html.Span(f"new advice after removing {c.get('target')}: ",
                           style={"fontWeight": "700", "color": "#64748b"})]
                + [html.Div("• " + str(n.get("action") or ""), style={"paddingLeft": "8px"})
                   for n in new],
                className="iv-new", style={"fontSize": "11px", "color": "#475569",
                                           "margin": "6px 0 0"}))
    return out


_STATUS = {"grounded": "#16a34a", "spurious": "#b45309", "escalated": "#b91c1c",
           "ignored_core": "#b91c1c", "correctly_ignored": "#64748b",
           "untested": "#94a3b8"}
_STATUS_MEANS = {
    "grounded": "a core hazard, and the advice eased when it was removed",
    "spurious": "not a core hazard, yet the advice eased when it was removed",
    "escalated": "removing it made the advice MORE alarmed (a red flag)",
    "ignored_core": "a core hazard, but the advice did not move",
    "correctly_ignored": "not a core hazard, and the advice did not move",
    "untested": "not removed yet"}
_SCENE_MEANS = {
    "grounded": "it declares the right hazard and the advice depends on it",
    "masquerade": "it declares the right hazard but the advice does not depend on it",
    "robust": "it declares the wrong hazard but the advice depends on the right one",
    "ungrounded": "it declares the wrong hazard and the advice does not depend on the right one",
    "spurious_grounding": "it declares the right hazard but the advice leans most on a wrong one",
    "pending": "the core hazard has not been removed yet",
    "incomplete": "the core hazard moved the advice; its competitor is not tested yet",
    "unreliable": "the try on the core hazard was not a clean test",
    "no_gt": "no ground truth"}
_DIST_MEANS = {
    "grounded": "the advice leans on the hazards in the proportion ground truth gives",
    "misproportioned": "the right hazards, the wrong emphasis",
    "masquerade": "the core hazard was removed and the advice barely moved",
    "spurious": "the advice leans on hazards ground truth never names",
    "pending": "no core hazard removed yet", "no_gt": "no ground truth"}
_READING = {"matches": "#16a34a", "said more than done": "#b45309",
            "done more than said": "#b45309", "more alarmed when removed": "#b91c1c",
            "not removed yet": "#94a3b8"}
_READING_MEANS = {
    "matches": "its share of what the model says and its share of what the advice "
               "depends on are close",
    "said more than done": "the model says it matters more than its advice depends on it",
    "done more than said": "the advice depends on it more than the model says it matters",
    "more alarmed when removed": "removing it made the answer more alarmed, not less",
    "not removed yet": "no try has removed this hazard"}
_SCORE_COL = {"does what it says": "#16a34a", "partly does what it says": "#b45309",
              "does not do what it says": "#b91c1c"}


def _bar(v: Any, col: str) -> Any:
    v = float(v) if isinstance(v, (int, float)) else 0.0
    return html.Div([
        html.Div(style={"width": f"{max(0.0, min(1.0, v)) * 100:.0f}%", "height": "7px",
                        "background": col, "borderRadius": "3px"}),
    ], title=f"{v:.2f}", style={"background": "#e2e8f0", "borderRadius": "3px",
                                "width": "70px", "display": "inline-block",
                                "verticalAlign": "middle"})


def new_advice(before: dict, tbl: dict) -> list[dict]:
    """Advice that is not in the original answer and shows up after a
    hazard was removed, grouped by the hazard it names: [{hazard, tries, of,
    in_original_scene}]. `in_original_scene` is True when that hazard was in
    the original picture and the original advice never named it as a cause:
    the model saw it and wrote nothing until something else was gone."""
    cols = [c for c in (tbl.get("columns") or []) if isinstance(c, dict)]
    seen = {str(o.get("object_id")) for o in (before.get("detected_objects") or [])
            if isinstance(o, dict)}
    named = {str(r.get("threat")) for r in (tbl.get("rows") or []) if isinstance(r, dict)}
    count: dict[str, int] = {}
    for c in cols:
        hazards = set()
        for rec in c.get("new") or []:
            q = rec.get("structured_reasoning") if isinstance(rec, dict) else None
            h = str((q or {}).get("threat") or "") if isinstance(q, dict) else ""
            if h:
                hazards.add(h)
        for h in hazards:
            count[h] = count.get(h, 0) + 1
    return [{"hazard": h, "tries": n, "of": len(cols),
             "in_original_scene": h in seen and h not in named}
            for h, n in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))]


def card_component(cmp: dict, tbl: dict, fresh: list[dict], syn: dict | None = None) -> list:
    """The causal-groundedness card: the takeaway, updated with every try.

    Its focus is DECLARED against MECHANISTIC (Sunny, 2026-11-16): what the
    model says matters, against what its advice depends on. Nothing on it
    needs ground truth, so it is the card a deployment can show. `syn` is
    given only when the "compare with ground truth" switch is on; it adds
    one line. The proof sits in the folds under the card, said once."""
    from agentic.ui_clean import ent_chip
    cmp = cmp if isinstance(cmp, dict) else {}
    lab = {"fontWeight": "700", "color": "#64748b"}
    line = {"fontSize": "11.5px", "margin": "2px 0", "lineHeight": "1.9"}
    score, reading = cmp.get("score"), str(cmp.get("reading") or "")
    col = _SCORE_COL.get(reading, "#64748b")
    cov = f"{cmp.get('tested', 0)} of {cmp.get('declared', 0)} declared hazards removed"
    rows: list = [html.Div([
        html.Span("DECLARED vs MECHANISTIC ",
                  style={"fontSize": "10px", "fontWeight": "800", "letterSpacing": ".08em",
                         "color": "#64748b"}),
        html.Span((f"{score:.2f} · {reading}" if isinstance(score, (int, float))
                   else "no score yet: remove a second hazard to compare"),
                  className="iv-score",
                  title="how far what the model says matters and what its advice depends "
                        "on agree, over the hazards removed so far (1.00 = the same shares)",
                  style={"fontSize": "15px" if isinstance(score, (int, float)) else "11.5px",
                         "fontWeight": "800", "color": col}),
    ])]
    if isinstance(score, (int, float)):
        rows.append(html.Div(html.Div(style={"width": f"{max(0, min(1, score)) * 100:.0f}%",
                                             "height": "6px", "background": col,
                                             "borderRadius": "3px"}),
                             style={"background": "#e2e8f0", "borderRadius": "3px",
                                    "margin": "4px 0 6px", "height": "6px"}))
    rows.append(html.Div(cov + (" · provisional" if cmp.get("provisional") else ""),
                         className="iv-coverage",
                         style={"fontSize": "10.5px", "color": "#94a3b8",
                                "marginBottom": "4px"}))

    def one(label: str, ids: list, colour: str, tail: str = "", cls: str = "") -> None:
        ids = [str(x) for x in ids if x]
        if ids:
            rows.append(html.Div(
                [html.Span(label + ": ", style=lab)] + [ent_chip(x, colour) for x in ids]
                + ([html.Span(tail, style={"color": "#64748b"})] if tail else []),
                className=cls, style=line))
    says, does = cmp.get("says_most"), cmp.get("does_most")
    if says and says == does:
        one("says and does most", [says], "#16a34a", cls="iv-most")
    else:
        one("says matters most", [says], "#334155", cls="iv-says")
        one("advice depends most on", [does], "#334155", cls="iv-does")
    one("advice depends least on", [cmp.get("does_least")], "#334155", cls="iv-least")
    one("said more than done", cmp.get("said_more") or [], "#b45309", cls="iv-said-more")
    one("done more than said", cmp.get("done_more") or [], "#b45309", cls="iv-done-more")
    one("more alarmed when removed", cmp.get("alarmed") or [], "#b91c1c", cls="iv-alarmed")

    verdicts = [str(r.get("verdict")) for r in (tbl.get("rows") or []) if isinstance(r, dict)]
    judged = [v for v in verdicts if v != "untested"]
    if verdicts:
        tied = sum(1 for v in judged if v == "grounded")
        rows.append(html.Div(
            [html.Span("advice tied to the hazard it names: ", style=lab),
             html.Span(f"{tied} of {len(judged)}" if judged else "none tested yet",
                       style={"fontWeight": "800",
                              "color": "#16a34a" if judged and tied == len(judged)
                              else "#b45309" if judged else "#94a3b8"})]
            + ([html.Span(f" · {len(verdicts) - len(judged)} not tested yet",
                          style={"color": "#94a3b8"})]
               if len(verdicts) > len(judged) else []),
            className="iv-tied",
            title="a recommendation is tied to its hazard when it is dropped once "
                  "that hazard is removed",
            style=line))
    # only advice about a hazard that was in the original scene and that the
    # original advice never acted on: the model saw it and wrote nothing
    # until something else was gone. Other new advice stays in the evidence.
    held = [f for f in (fresh or []) if isinstance(f, dict) and f.get("in_original_scene")]
    if held:
        kids: list = [html.Span("new advice after removing a hazard: ", style=lab)]
        for f in held:
            kids += [ent_chip(str(f.get("hazard")), "#b45309"),
                     html.Span(f"({f.get('tries')} of {f.get('of')} tries) ",
                               style={"color": "#64748b"})]
        kids.append(html.Div("in the original scene, and the original advice did not "
                             "act on it", style={"fontSize": "10.5px", "color": "#94a3b8",
                                                 "lineHeight": "1.3"}))
        rows.append(html.Div(kids, className="iv-new-advice", style=line))
    one("not removed yet", cmp.get("not_removed") or [], "#94a3b8", cls="iv-not-removed")
    if isinstance(syn, dict):
        if syn.get("available"):
            d = syn.get("distribution") or {}
            v = str(d.get("verdict"))
            c = d.get("coverage") or []
            rows.append(html.Div(
                [html.Span("against ground truth: ", style=lab),
                 html.Span(f"{d.get('overlap')} · {v.replace('_', ' ')}",
                           style={"fontWeight": "800"}),
                 html.Span(" — " + _DIST_MEANS.get(v, "")
                           + (f" · {c[0]} of {c[1]} core hazards removed" if len(c) == 2 else ""),
                           style={"color": "#64748b"})],
                className="iv-gt-line", style=line))
        else:
            rows.append(html.Div(
                "against ground truth: not available — "
                + str(syn.get("why") or "no ground truth for this scene"),
                className="iv-gt-line", style={**line, "color": "#94a3b8"}))
    rows.append(html.Div("one ask per try · agreeing with itself is not the same as "
                         "being right",
                         style={"fontSize": "10px", "color": "#94a3b8", "marginTop": "4px"}))
    return [html.Div(rows, className="unc-panel iv-card",
                     style={"borderColor": col, "padding": "8px 10px"})]


def hazards_component(cmp: dict, syn: dict | None = None) -> list:
    """Evidence: each hazard, what the model says against what the advice
    does. With the ground-truth switch on (`syn`), a "should" bar and the
    ground-truth status are added."""
    from agentic.ui_clean import ent_chip
    gt_on = isinstance(syn, dict) and bool(syn.get("available"))
    dist = (syn.get("distribution") or {}) if gt_on else {}
    gt_d = dist.get("gt_dist") or {}
    status = {str(h.get("object_id")): h for h in (syn.get("hazards") or [])} if gt_on else {}
    head = ["HAZARD", "SAYS", "DOES"] + (["SHOULD (GROUND TRUTH)"] if gt_on else []) \
        + ["READING"] + (["AGAINST GROUND TRUTH"] if gt_on else [])
    body = []
    rows_in = cmp.get("hazards") if isinstance(cmp, dict) else None
    for r in rows_in if isinstance(rows_in, list) else []:
        if not isinstance(r, dict):
            continue
        oid, reading = str(r.get("object_id")), str(r.get("reading"))
        tds = [html.Td(ent_chip(oid, "#ef4444"), style=_TD),
               html.Td(_bar(r.get("says"), "#334155"), style=_TD),
               html.Td(_bar(r.get("does"), "#16a34a") if r.get("does") is not None
                       else html.Span("—", style={"color": "#cbd5e1"}), style=_TD)]
        if gt_on:
            tds.append(html.Td(_bar(gt_d.get(oid), "#7c3aed"), style=_TD))
        tds.append(html.Td(reading, className="iv-reading", title=_READING_MEANS.get(reading, ""),
                           style={**_TD, "fontWeight": "800",
                                  "color": _READING.get(reading, "#64748b")}))
        if gt_on:
            st = str((status.get(oid) or {}).get("status") or "")
            core = (status.get(oid) or {}).get("is_core")
            tds.append(html.Td(
                (st.replace("_", " ") + (" · core" if core else "")) if st else "not in ground truth",
                className="iv-hazard-status", title=_STATUS_MEANS.get(st, ""),
                style={**_TD, "fontWeight": "700", "color": _STATUS.get(st, "#94a3b8")}))
        body.append(html.Tr(tds, className="iv-hazard"))
    if not body:
        return []
    out: list = [
        html.Div("says: the share of each hazard in the model's own graph of the scene, "
                 "arrows weighed by what their chain reaches · does: the share of how far "
                 "the answer eased when it was removed",
                 style={"fontSize": "10px", "color": "#94a3b8", "margin": "2px 0 4px"}),
        html.Table([html.Thead(html.Tr([html.Th(x, style=_TH) for x in head])),
                    html.Tbody(body)], className="iv-hazards",
                   style={"borderCollapse": "collapse", "width": "100%"})]
    if gt_on:
        gt = syn.get("gt") or {}
        cell = str(syn.get("cell"))
        out.append(html.Div(
            f"ground truth: {gt.get('status')} · {gt.get('image_filename')}"
            + (" (the same picture, matched by its content)"
               if gt.get("matched_by") == "content" else "")
            + f" · the scene against ground truth: {cell.replace('_', ' ')}"
            + (f" — {_SCENE_MEANS.get(cell, '')}" if _SCENE_MEANS.get(cell) else ""),
            className="iv-gt", style={"fontSize": "10px", "color": "#94a3b8",
                                      "marginTop": "4px"}))
    return out


def _fold(title: str, kids: list, open_: bool = False, cls: str = "") -> Any:
    return html.Details([
        html.Summary(title, style={"fontSize": "10.5px", "fontWeight": "700",
                                   "color": "#475569", "cursor": "pointer",
                                   "marginTop": "6px"}),
        html.Div(kids, style={"padding": "4px 0 2px 4px"})],
        open=open_, className=("iv-evidence " + cls).strip())


def result_component(base_dir: Path | None, runs_dir: Path | None,
                     active: dict | None = None, with_gt: bool = False) -> list:
    """Everything shown above the form: the card (the takeaway), then the
    evidence in folds, said once: each recommendation by removed hazard,
    each hazard says against does, and the tries. `with_gt` is the "compare
    with ground truth" switch (calibration only; off in deployment)."""
    from agentic import chain4, declared4, grounding4
    tries = load_tries(base_dir)
    out: list = []
    measured = [t for t in tries if t.get("status") == "measured"
                and (t.get("summary") or {}).get("fair") is not False
                and t.get("after_run") and runs_dir]
    lines = status_component(tries, active)
    running = [x for x in lines if "iv-active" in str(getattr(x, "className", ""))]
    if not (measured and base_dir):
        return lines
    try:
        before = iv4.result_from_run_dir(base_dir)
        pairs = [(t, iv4.result_from_run_dir(Path(runs_dir) / str(t["after_run"])))
                 for t in measured]
        tbl = grounding4.table(before, pairs)
        for col, (t, after) in zip(tbl["columns"], pairs):
            try:
                col["chain"] = chain4.direction(before, after)
            except Exception:
                col["chain"] = None             # the table still stands without it
            col["signed"] = chain4.signed_shift(t.get("signals"), col["chain"])
        cmp = declared4.compare(before, pairs)
        syn = iv4.synthesis(before, pairs) if with_gt else None
        out += card_component(cmp, tbl, new_advice(before, tbl), syn)
        out.append(_fold("evidence: each recommendation, by removed hazard",
                         table_component(tbl, runs_dir),
                         open_=len(tbl["columns"]) < 2, cls="by-recommendation"))
        out.append(_fold("evidence: each hazard, says against does",
                         hazards_component(cmp, syn), cls="by-hazard"))
    except Exception as exc:                    # never lose the try lines to a bad card
        out.append(html.Div(f"the card could not be built: {exc}",
                            className="ticket-empty"))
    out.append(_fold("the tries, one line each",
                     [x for x in lines if x not in running], cls="tries"))
    return out + running


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
    return out


def summary_sentence(sm: Any) -> str:
    """The "rests on" half of the headline, from scene_summary(): does the
    advice depend on what the model says matters. Short."""
    if not isinstance(sm, dict) or not sm.get("n"):
        return ""
    reading = str(sm.get("reading") or "")
    said = {"does what it says": "Advice depends on what it says matters",
            "partly does what it says": "Advice partly depends on what it says matters",
            "does not do what it says": "Advice does not depend on what it says matters",
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
                "whether the advice moves")
    n = sm["n"]
    head = (f"{n} hazard{'s' if n != 1 else ''} removed "
            f"({', '.join(sm.get('targets') or [])})")
    score = sm.get("score")
    if not isinstance(score, (int, float)):
        return head + " · no score yet: remove a second hazard to compare"
    return (head + f" · declared vs mechanistic: {score:.2f}, {sm.get('reading')}"
            + (f" · {sm.get('tested')} of {sm.get('declared')} declared hazards removed"
               if sm.get("provisional") else ""))


def try_line(t: dict) -> str:
    """One plain sentence for a saved try."""
    head = f"#{t.get('index', '?')} removed {t.get('target')} ({MODALITY_LABEL.get(str(t.get('modality')), t.get('modality'))})"
    if t.get("status") != "measured":
        return f"{head}: not measured — {t.get('why') or 'unknown reason'}"
    s = t.get("summary") or {}
    if s.get("fair") is False:
        return f"{head}: not a fair test — the input was not changed"
    if s.get("moved") is None:
        return f"{head}: could not be judged"
    shift = s.get("content_shift")
    shift_txt = f" (shift {shift:.2f})" if isinstance(shift, (int, float)) else ""
    lvl = s.get("level_delta")
    lvl_txt = f", danger level {lvl:+d}" if isinstance(lvl, int) and lvl else ""
    return (f"{head}: the advice " + ("MOVED" if s["moved"] else "did NOT move")
            + shift_txt + lvl_txt)


def status_component(tries: list[dict], active: dict | None) -> list:
    """The tries so far, newest last, and the running one."""
    out: list = []
    for t in tries or []:
        s = t.get("summary") or {}
        moved = s.get("moved") if t.get("status") == "measured" else None
        col = "#16a34a" if moved else "#b91c1c" if moved is False else "#64748b"
        out.append(html.Div(try_line(t), className="iv-try",
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
