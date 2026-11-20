"""How far along a live run is (Sunny, 2026-11-20: "a progress bar... if
progress is not possible then how much time has passed").

A run writes an event each time something starts or ends. Steps are very
unequal (naming objects takes seconds, writing the recommendations takes
minutes), so counting steps would make a bar that jumps and then stands
still. Instead the bar is sized by TIME: the last finished run with the
same settings says how long each stretch took, and the live run is placed
on that timeline.

    reference run:   perceive 51s | ground 12s | ... | recommend 213s | ...
    live run:        reached "recommend" 20s ago  ->  bar at that point + 20s

With no finished run of the same settings there is nothing to learn from:
the bar has no fill and only the step and the time passed are shown.
Inside one long model call nothing is reported, so the bar creeps on by
the clock up to the next mark and waits there.

Pure code: no model call, no prompt. Never raises on a torn file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dash import html

DONE_EVENT = "stage4_result"          # the last event of a finished run
_SCAN_EVERY = 60.0                    # seconds between looks at the run folder
_cache: dict[tuple, tuple[float, Any]] = {}

# What is going on AFTER each mark, in plain words. Checked from the end:
# the last one reached names the step.
STEPS: list[tuple[tuple[str, str], str]] = [
    (("stage_started", "Perceive"), "naming what is in the picture"),
    (("stage_started", "Repair"), "checking the list of objects"),
    (("stage_started", "Ground"), "finding each object in the picture"),
    (("stage_started", "Mask"), "outlining each object"),
    (("stage_started", "Assemble"), "putting the scene together"),
    (("stage_started", "assess"), "assessing the scene"),
    (("assess_verdict", ""), "checking the assessment"),
    (("stage_started", "recommend"), "writing the recommendations"),
    (("recommendations_ready", ""), "drawing its own graph of the scene"),
    (("graph_b_built", ""), "scoring the recommendations"),
    (("stage_done", "recommend"), "finishing"),
]


def key(ev: Any) -> tuple[str, str]:
    """The mark an event stands for: its type and, for a stage event, the stage."""
    if not isinstance(ev, dict):
        return ("", "")
    t = ev.get("type")
    s = ev.get("stage")
    return (t if isinstance(t, str) else "", s if isinstance(s, str) else "")


def _t(ev: Any) -> float | None:
    t = ev.get("t") if isinstance(ev, dict) else None
    return float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None


def settings_of(events: Any) -> str:
    """The run's settings as one comparable string ('' when not recorded)."""
    for ev in (events if isinstance(events, list) else [])[:6]:
        if isinstance(ev, dict) and ev.get("type") == "run_settings":
            try:
                return json.dumps({k: v for k, v in ev.items() if k not in ("t", "type")},
                                  sort_keys=True, default=str)
            except Exception:
                return ""
    return ""


def start_of(events: Any) -> float | None:
    """When the run began: its first timestamped event."""
    for ev in events if isinstance(events, list) else []:
        if _t(ev) is not None:
            return _t(ev)
    return None


def timeline(events: Any) -> dict:
    """{"marks": {mark: seconds from the start when it was first reached},
    "total": seconds, "finished": bool} for one run's events."""
    out: dict = {"marks": {}, "total": 0.0, "finished": False}
    t0 = start_of(events)
    if t0 is None:
        return out
    last = t0
    for ev in events:
        t = _t(ev)
        if t is None:
            continue
        last = max(last, t)
        out["marks"].setdefault(key(ev), max(0.0, t - t0))
        if isinstance(ev, dict) and ev.get("type") == DONE_EVENT:
            out["finished"] = True
    out["total"] = max(0.0, last - t0)
    return out


def _read(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    try:
        with Path(path).open() as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    out.append(ev)
    except OSError:
        pass
    return out


def reference(runs_dir: Any, settings: str, exclude: Any = (), now: float | None = None) -> Any:
    """The timeline of the newest FINISHED run with the same settings, or
    None. The folder is looked at once a minute at most."""
    if not settings or not runs_dir:
        return None
    now = time.time() if now is None else now
    ck = (str(runs_dir), settings, tuple(sorted(str(x) for x in (exclude or ()))))
    hit = _cache.get(ck)
    if hit and now - hit[0] < _SCAN_EVERY:
        return hit[1]
    found = None
    try:
        files = sorted(Path(runs_dir).glob("*/events.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        files = []
    skip = {str(x) for x in (exclude or ())}
    for p in files:
        if p.parent.name in skip:
            continue
        if settings_of(_read(p, 6)) != settings:
            continue
        tl = timeline(_read(p))
        if tl["finished"] and tl["total"] > 0:
            found = tl
            break
    _cache[ck] = (now, found)
    return found


def step_label(events: Any) -> str:
    """What the run is doing now, in plain words."""
    seen = {key(ev) for ev in (events if isinstance(events, list) else [])}
    for mark, label in reversed(STEPS):
        if mark in seen:
            return label
    return "starting"


def estimate(events: Any, ref: Any = None, now: float | None = None,
             done: bool = False) -> dict:
    """{"elapsed": seconds, "step": words, "fraction": 0..1 or None,
    "left": seconds or None, "done": bool}. `fraction` and `left` are None
    when there is no reference run to size the bar by."""
    now = time.time() if now is None else now
    events = events if isinstance(events, list) else []
    t0 = start_of(events)
    elapsed = max(0.0, now - t0) if t0 is not None else 0.0
    out = {"elapsed": elapsed, "step": step_label(events), "fraction": None,
           "left": None, "done": bool(done)}
    if done:
        last = max((_t(e) for e in events if _t(e) is not None), default=t0)
        out.update(fraction=1.0, left=0.0, step="done",
                   elapsed=max(0.0, (last or 0.0) - (t0 or 0.0)))
        return out
    ref = ref if isinstance(ref, dict) else {}
    marks = ref.get("marks") if isinstance(ref.get("marks"), dict) else {}
    total = ref.get("total")
    if t0 is None or not marks or not isinstance(total, (int, float)) or total <= 0:
        return out
    # the furthest mark of the reference this run has reached, and when
    at_ref, reached = 0.0, t0
    for ev in events:
        k, t = key(ev), _t(ev)
        r = marks.get(k)
        if t is not None and isinstance(r, (int, float)) and r >= at_ref:
            if r > at_ref or t < reached:
                at_ref, reached = float(r), t
    later = [float(v) for v in marks.values() if isinstance(v, (int, float)) and v > at_ref]
    nxt = min(later) if later else float(total)
    # creep on by the clock, but never past the next mark (it waits there)
    pos = at_ref + min(max(0.0, now - reached), max(0.0, nxt - at_ref) * 0.97)
    out["fraction"] = max(0.0, min(0.99, pos / float(total)))
    out["left"] = max(0.0, float(total) - pos)
    return out


def clock(seconds: Any) -> str:
    """m:ss (h:mm:ss past an hour)."""
    try:
        s = max(0, int(seconds))
    except (TypeError, ValueError):
        s = 0
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def about(seconds: Any) -> str:
    """'about 4 min left' / 'under a minute left'."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return ""
    if s < 60:
        return "under a minute left"
    return f"about {int(round(s / 60))} min left"


def component(est: Any, what: str = "") -> Any:
    """The bar with its line of words. `what` leads the line (for a try:
    what was removed)."""
    est = est if isinstance(est, dict) else {}
    frac = est.get("fraction")
    known = isinstance(frac, (int, float)) and not isinstance(frac, bool)
    pct = int(round(100 * max(0.0, min(1.0, float(frac))))) if known else 0
    words = [w for w in (what, str(est.get("step") or ""),
                         clock(est.get("elapsed")) + " passed",
                         about(est.get("left")) if known and not est.get("done") else "",
                         "" if known else "no earlier run with these settings to time it by")
             if w]
    fill = {"height": "100%", "borderRadius": "5px", "background": "#2563eb",
            "width": f"{pct}%", "transition": "width .7s linear"}
    if not known:
        # nothing to size it by: a pale full bar, the words carry the news
        fill.update(width="100%", background="#bfdbfe")
    return html.Div([
        html.Div(html.Div(className="run-progress-fill", style=fill),
                 className="run-progress-track",
                 **{"data-known": "yes" if known else "no"},
                 style={"height": "8px", "borderRadius": "5px", "background": "#e2e8f0",
                        "overflow": "hidden"}),
        html.Div((f"{pct}% · " if known else "") + " · ".join(words),
                 className="run-progress-words",
                 style={"fontSize": "11.5px", "fontWeight": "600", "color": "#334155",
                        "marginTop": "3px"}),
    ], className="run-progress", style={"margin": "0 0 8px"})
