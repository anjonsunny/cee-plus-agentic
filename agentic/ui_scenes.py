"""The page across scenes: the latest finished run of every scene in one
table, and the patterns that only show when the runs sit side by side.

Sunny, 2026-11-13 ("build it"). During the A→H clean-view pass every
cross-scene pattern (house_2 dropped under "only one crew" on A and G, the
dispatch check skipped quietly on B and F) was found by reading eight
records by hand. This page does that reading.

Pure: it reads events.jsonl files and returns rows; the Dash wiring lives
in ui.py. A row is built with the same `derive` and the same headline the
single-run page uses, so the two can never disagree. Read-only and
dev-side: nothing here feeds a prompt or a score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from dash import html

from agentic import ui_clean

HEAD_LINES = 4      # run_started is the 1st or 2nd line; look a little further


def run_head(path: Path) -> dict:
    """The run's `run_started` event, from the first few lines only (the
    settings event may come before it). {} when unreadable."""
    try:
        with Path(path).open() as f:
            for _ in range(HEAD_LINES):
                line = f.readline()
                if not line:
                    break
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict) and ev.get("type") == "run_started":
                    return ev
    except OSError:
        pass
    return {}


def _events(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue                    # a torn last line of a live run
            if isinstance(ev, dict):
                out.append(ev)
    except OSError:
        pass
    return out


def row_of(path: Path, derive: Callable[[list], dict]) -> dict | None:
    """One table row for a finished run, or None when the run has no
    recommendation result (still running, failed, or perception only)."""
    events = _events(path)
    d = derive(events)
    s4 = d.get("stage4")
    if not isinstance(s4, dict) or not s4:
        return None
    verdict = ((d.get("assess") or {}).get("verdict") or {})
    trust = s4.get("trust") if isinstance(s4.get("trust"), dict) else {}
    try:
        tickets = [t for t in (ui_clean._register_of(s4).get("pathology") or [])
                   if isinstance(t, dict)]
    except Exception:
        tickets = []
    tone, _parts = ui_clean.headline_parts(s4, d)
    ts = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    probes = s4.get("env_probes") if isinstance(s4.get("env_probes"), dict) else {}
    ran = {k: v for k, v in probes.items()
           if isinstance(v, dict) and not v.get("skipped") and not v.get("error")}
    asked_env = bool((d.get("reasks") or {}).get("env")
                     or (d.get("run_settings") or {}).get("env"))
    unaddressed = sorted({str(x) for e in (trust.get("singular_errors") or [])
                          if isinstance(e, dict) and e.get("id") == "hazard_unaddressed"
                          for x in (e.get("entities") or [])})
    hidden = []
    for tk in tickets:
        seg = [x.strip() for x in str(tk.get("subtype") or "").split("·")]
        if len(seg) >= 3 and seg[0] == "environment":
            hidden.append({"condition": seg[1], "what": seg[-1],
                           "dropped": [str(x) for x in (tk.get("dropped") or [])],
                           "k": tk.get("hidden_in"), "n": tk.get("n_answers")})
    image = str(d.get("image_name") or "")
    return {
        "scene": image.rsplit(".", 1)[0] or Path(path).parent.name,
        "run": Path(path).parent.name, "path": str(path),
        "when": max(ts) if ts else None,
        "minutes": round((max(ts) - min(ts)) / 60, 1) if len(ts) > 1 else None,
        "disaster": verdict.get("scenario") == "Yes" if verdict else None,
        "disaster_type": verdict.get("disaster_type"),
        "level": verdict.get("level"),
        "trust": trust.get("score"), "band": str(trust.get("band") or ""),
        "n_tickets": len(tickets),
        "tone": tone, "headline": ui_clean.headline_text(s4, d),
        "unaddressed": unaddressed, "hidden": hidden,
        "env_asked": asked_env,
        "env_skipped": d.get("stage4_env_skipped") if asked_env else None,
        "env_clean": bool(asked_env and ran and not hidden
                          and not d.get("stage4_env_skipped")),
    }


def scene_rows(runs_dir: Path, derive: Callable[[list], dict]) -> list[dict]:
    """The latest FINISHED run of each scene, sorted by scene name. A scene
    is its image name without the extension. Runs are walked newest first
    and a run is parsed in full only while its scene has no row yet."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    paths = sorted(runs_dir.glob("*/events.jsonl"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    rows: dict[str, dict] = {}
    for p in paths:
        if (p.parent / "after_of.json").exists():
            continue                        # the edited side of an intervention
        image = str(run_head(p).get("image_name") or "")
        scene = image.rsplit(".", 1)[0]
        if not scene or scene in rows:
            continue
        try:
            row = row_of(p, derive)
        except Exception:
            row = None                      # one broken record never empties the page
        if row is not None:
            rows[scene] = row
    return [rows[k] for k in sorted(rows)]


def patterns(rows: list[dict]) -> list[dict]:
    """What repeats across scenes: [{"text", "scenes"}], most scenes first.
    Each pattern is a plain fact about the rows, not a judgement."""
    found: dict[str, list[str]] = {}

    def add(text: str, scene: str) -> None:
        if scene not in found.setdefault(text, []):
            found[text].append(scene)
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        s = str(r.get("scene") or "")
        for h in r.get("hidden") or []:
            if isinstance(h, dict):
                ids = ", ".join(h.get("dropped") or [])
                add(f'told "{h.get("condition")}", stopped acting on '
                    + (ids or "something it acted on before"), s)
        if r.get("unaddressed"):
            add("a danger it named is left without action", s)
        if r.get("env_skipped"):
            add(f"concealment was not checked: {r['env_skipped']}", s)
        if r.get("env_clean"):
            add("every dispatch line asked came back clean", s)
        if r.get("disaster") is False:
            add("no disaster seen", s)
    out = [{"text": k, "scenes": v} for k, v in found.items()]
    return sorted(out, key=lambda p: (-len(p["scenes"]), p["text"]))


_TONE = {"high": "#16a34a", "moderate": "#b45309", "low": "#b91c1c"}
_TH = {"textAlign": "left", "fontSize": "10px", "letterSpacing": ".06em",
       "color": "#64748b", "padding": "4px 8px", "borderBottom": "1px solid #cbd5e1"}
_TD = {"fontSize": "11.5px", "padding": "5px 8px", "borderBottom": "1px solid #e2e8f0",
       "verticalAlign": "top"}


def table(rows: list[dict], open_id: Callable[[str], Any]) -> list:
    """The table and the patterns as Dash components. `open_id(path)` gives
    the id of a row's "open" button (ui.py owns the callback)."""
    if not rows:
        return [html.Div("no finished run yet: a scene shows here once its "
                         "recommendations are written", className="ticket-empty")]
    import datetime as _dt
    head = html.Tr([html.Th(h, style=_TH) for h in
                    ("SCENE", "VERDICT", "TRUST", "TICKETS", "MIN", "HEADLINE", "RUN", "")])
    body = []
    for r in rows:
        col = _TONE.get(str(r.get("tone")), "#64748b")
        if r.get("disaster") is None:
            verdict = "not assessed"
        elif r["disaster"]:
            verdict = f"{r.get('disaster_type') or 'disaster'} · level {r.get('level')}"
        else:
            verdict = "no disaster"
        when = (_dt.datetime.fromtimestamp(float(r["when"])).strftime("%b %d %H:%M")
                if r.get("when") else "")
        body.append(html.Tr([
            html.Td(str(r.get("scene")), style={**_TD, "fontWeight": "800"}),
            html.Td(verdict, style=_TD),
            html.Td(f"{r['trust']} · {r.get('band')}" if r.get("trust") is not None
                    else "not scored", style={**_TD, "color": col, "fontWeight": "700",
                                              "whiteSpace": "nowrap"}),
            html.Td(str(r.get("n_tickets") or 0),
                    style={**_TD, "fontWeight": "700",
                           "color": "#be123c" if r.get("n_tickets") else "#94a3b8"}),
            html.Td("" if r.get("minutes") is None else str(r["minutes"]), style=_TD),
            html.Td(str(r.get("headline") or ""), style={**_TD, "color": col}),
            html.Td(f"{when} · {r.get('run')}",
                    style={**_TD, "color": "#94a3b8", "fontSize": "10px",
                           "whiteSpace": "nowrap"}),
            html.Td(html.Button("open", id=open_id(str(r.get("path"))), n_clicks=0,
                                className="scene-open",
                                style={"fontSize": "10px", "cursor": "pointer"}),
                    style=_TD),
        ], className=f"scene-row tone-{r.get('tone')}"))
    out: list = [html.Table([html.Thead(head), html.Tbody(body)],
                            className="scene-table",
                            style={"borderCollapse": "collapse", "width": "100%"})]
    pats = patterns(rows)
    if pats:
        out.append(html.Div("SEEN ACROSS SCENES",
                            style={"fontSize": "10px", "fontWeight": "800",
                                   "letterSpacing": ".08em", "color": "#64748b",
                                   "margin": "10px 0 4px"}))
        out += [html.Div([html.Span(f"{len(p['scenes'])} · ",
                                    style={"fontWeight": "800", "color": "#334155"}),
                          html.Span(p["text"] + ": "),
                          html.Span(", ".join(p["scenes"]), style={"fontWeight": "700"})],
                         className="scene-pattern",
                         style={"fontSize": "11.5px", "color": "#475569",
                                "margin": "2px 0"})
                for p in pats]
    out.append(html.Div("the latest finished run of each scene · built from the "
                        "saved records, nothing is re-run",
                        style={"fontSize": "10px", "color": "#94a3b8", "marginTop": "8px"}))
    return out
