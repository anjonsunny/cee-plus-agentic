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


_CELL = {"stays": ("stays", "#64748b"), "gone": ("gone", "#0f172a"),
         "changed": ("changed", "#b45309"), "target removed": ("target removed", "#94a3b8")}
_VERDICT = {"grounded": "#16a34a", "decoration": "#b91c1c",
            "hidden dependence": "#b45309", "mixed": "#b45309",
            "not tested yet": "#94a3b8"}
_TH = {"textAlign": "left", "fontSize": "10px", "color": "#64748b", "padding": "4px 6px",
       "borderBottom": "1px solid #cbd5e1", "verticalAlign": "bottom"}
_TD = {"fontSize": "11px", "padding": "4px 6px", "borderBottom": "1px solid #e2e8f0",
       "verticalAlign": "top"}


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
            + ([html.Div("gone with it: " + ", ".join(also),
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
            word, col = _CELL.get(str(c.get("state")), (str(c.get("state")), "#64748b"))
            after = c.get("after") if isinstance(c.get("after"), dict) else {}
            extra = " · action changed" if c.get("action_changed") else ""
            tds.append(html.Td(word + extra, className=f"iv-cell {word.split()[0]}",
                               title=("after: " + str(after.get("action") or ""))
                               if after else "no matching advice in the edited run",
                               style={**_TD, "color": col, "fontWeight": "700"}))
        v = str(r.get("verdict"))
        tds.append(html.Td(v, className="iv-verdict",
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
    out: list = [html.Table([html.Thead(html.Tr(head)), html.Tbody(body + [html.Tr(foot)])],
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


def result_component(base_dir: Path | None, runs_dir: Path | None,
                     active: dict | None = None) -> list:
    """Everything shown above the form: the grounding table over the
    measured tries, one line per try, and the running one."""
    from agentic import grounding4
    tries = load_tries(base_dir)
    out: list = []
    measured = [t for t in tries if t.get("status") == "measured"
                and (t.get("summary") or {}).get("fair") is not False
                and t.get("after_run") and runs_dir]
    if measured and base_dir:
        try:
            tbl = grounding4.table(
                iv4.result_from_run_dir(base_dir),
                [(t, iv4.result_from_run_dir(Path(runs_dir) / str(t["after_run"])))
                 for t in measured])
            out += table_component(tbl, runs_dir)
        except Exception as exc:                # never lose the try lines to a bad table
            out.append(html.Div(f"the table could not be built: {exc}",
                                className="ticket-empty"))
    lines = status_component(tries, active)
    if measured:
        out.append(html.Details([
            html.Summary("the tries, one line each",
                         style={"fontSize": "10px", "color": "#64748b",
                                "cursor": "pointer", "marginTop": "6px"}),
            html.Div([x for x in lines if "iv-active" not in str(getattr(x, "className", ""))])]))
        out += [x for x in lines if "iv-active" in str(getattr(x, "className", ""))]
    else:
        out += lines
    return out


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
