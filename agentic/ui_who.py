"""WHO IS WHO: the pictures of a run and its tries side by side, the same
object in the same colour in every picture (Sunny, 2026-11-20: "what if we
color code the same objects and show two side by side image... and
trajectory can also do that later").

The recommendations table compares two runs by an object's name, so it is
only right when the framework matched the objects right (agentic/align4.py).
Identity is checked by looking, not by reading a list: this strip is where
that is done.

    before        #1 house_1 dealt with     #2 car_1 dealt with
    [picture]     [picture]                 [picture]
                  all sure                  1 not sure

Colour means IDENTITY here and nothing else (on the main picture it means
threat / at risk; the two are kept apart). Only the threats and those at
risk are shown, plus whatever a try was aimed at.
A match that is a guess has a dashed outline; a new object is tagged "new".

Pure drawing: no model call. The pictures are made once and kept in memory.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from dash import html

from agentic import align4, intervention4 as iv4

TITLE = "CAUSAL GROUNDEDNESS: WHO IS WHO"
WIDTH = 460                                   # pixels each picture is drawn at
# distinct, readable on photos; identity only
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
           "#800000", "#aaffc3", "#808000", "#000075"]
GREY = "#94a3b8"
_cache: dict[tuple, str] = {}


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _matter(result: Any) -> set[str]:
    """The threats and the at-risk of ONE run (Sunny, 2026-11-20: "for
    picture... only show threats and risks/victims"): what harms (its
    threats and the hazard each recommendation names) and who or what the
    assessment put at risk. A run with no at-risk list falls back to the
    things its arrows end at."""
    r = iv4.clean_result(result)
    out = set(iv4.hazards_of(r))
    risk = {str(x.get("object_id") or "") for x in (r.get("at_risk") or [])
            if isinstance(x, dict)}
    if not risk and not isinstance((result or {}).get("at_risk"), list):
        risk = {v["object_id"] for v in iv4.victims_of(r)}
    return {x for x in out | risk if x}


def important(before: Any, pairs: Any) -> list[str]:
    """The objects that get a colour, in the order the run lists them: its
    threats, its at-risk, and whatever a try was aimed at."""
    try:
        r = iv4.clean_result(before)
        ids = [str(o.get("object_id") or "") for o in r["detected_objects"]]
        tried = {str((t if isinstance(t, dict) else {}).get("target") or "")
                 for t, _a in (pairs or [])}
        want = _matter(before) | tried
        return [i for i in ids if i and i in want]
    except Exception:
        return []


def colours(before: Any, pairs: Any) -> dict[str, str]:
    """{object: colour}, the same in every picture of the strip."""
    return {oid: PALETTE[i % len(PALETTE)] for i, oid in enumerate(important(before, pairs))}


def draw(result: Any, colour_of: dict[str, str], unsure: Any = (), new: Any = (),
         its_name: Any = None) -> str | None:
    """One picture as a data URL: each coloured object tinted over its own
    shape, outlined and named. `its_name` {name here: the run's own name}
    adds the run's own name in brackets. None when the picture cannot be read."""
    try:
        from PIL import Image, ImageDraw
        r = result if isinstance(result, dict) else {}
        folder, name = r.get("run_dir"), r.get("scene_image") or r.get("image_filename")
        if not folder or not name:
            return None
        path = Path(str(folder)) / str(name)
        unsure, new = set(unsure or ()), set(new or ())
        its_name = its_name if isinstance(its_name, dict) else {}
        objs = [o for o in (r.get("detected_objects") or []) if isinstance(o, dict)]
        key = (str(path), path.stat().st_mtime, tuple(sorted(colour_of.items())),
               tuple(sorted(unsure)), tuple(sorted(new)), tuple(sorted(its_name.items())),
               tuple(str(o.get("object_id")) for o in objs))
        if key in _cache:
            return _cache[key]
        img = Image.open(path).convert("RGBA")
        scale = WIDTH / float(img.width)
        size = (WIDTH, max(1, int(round(img.height * scale))))
        img = img.resize(size)
        next_new = len(colour_of)
        todo = []
        for o in objs:
            oid = str(o.get("object_id") or "")
            if oid in colour_of:
                todo.append((o, oid, colour_of[oid]))
            elif oid in new:
                todo.append((o, oid, PALETTE[next_new % len(PALETTE)]))
                next_new += 1
        # the tints first, then the outlines and names on top
        for o, _oid, col in todo:
            try:
                mask = Image.open(str(o.get("mask_path"))).convert("L").resize(size)
                tint = Image.new("RGBA", size, _rgb(col) + (0,))
                tint.putalpha(mask.point(lambda v: 85 if v > 127 else 0))
                img = Image.alpha_composite(img, tint)
            except Exception:
                pass                          # no saved shape: the outline alone
        pen = ImageDraw.Draw(img)
        for o, oid, col in todo:
            box = o.get("bbox")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                x0, y0, x1, y1 = (float(v) * scale for v in box)
            except (TypeError, ValueError):
                continue
            if oid in unsure:                 # a guess: dashed
                step = 8
                for x in range(int(x0), int(x1), step * 2):
                    pen.line([(x, y0), (min(x + step, x1), y0)], fill=col, width=2)
                    pen.line([(x, y1), (min(x + step, x1), y1)], fill=col, width=2)
                for y in range(int(y0), int(y1), step * 2):
                    pen.line([(x0, y), (x0, min(y + step, y1))], fill=col, width=2)
                    pen.line([(x1, y), (x1, min(y + step, y1))], fill=col, width=2)
            else:
                pen.rectangle([x0, y0, x1, y1], outline=col, width=2)
            text = oid + (f" ({its_name[oid]})" if its_name.get(oid) else "") \
                + (" ?" if oid in unsure else "") + (" new" if oid in new else "")
            tw = pen.textlength(text)
            ty = max(0, y0 - 12)
            pen.rectangle([x0, ty, x0 + tw + 6, ty + 12], fill=col)
            pen.text((x0 + 3, ty), text, fill="white" if sum(_rgb(col)) < 420 else "black")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=82)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        _cache[key] = url
        return url
    except Exception:
        return None


def _its_names(after: Any) -> dict[str, str]:
    """{name used here: the edited run's own name}, where they differ."""
    ren = after.get("renamed") if isinstance(after, dict) else None
    return {here: its for its, here in (ren if isinstance(ren, dict) else {}).items()
            if isinstance(its, str) and isinstance(here, str)}


def says(after: Any) -> str:
    """'all sure' / '1 not sure', for one try."""
    s = align4.summary(after.get("tracks") if isinstance(after, dict) else None)
    return "all sure" if not s["not sure"] else f"{s['not sure']} not sure"


def not_sure(pairs: Any) -> int:
    """How many matches over all tries are a guess."""
    n = 0
    for _t, after in pairs or []:
        n += align4.summary(after.get("tracks") if isinstance(after, dict) else None)["not sure"]
    return n


def component(before: Any, pairs: Any, title_style: dict | None = None) -> list:
    """The fold: the original picture, then one picture per try. [] when
    there is no try or no picture to show."""
    from agentic import ui_intervene
    pairs = [(t, a) for t, a in (pairs or []) if isinstance(a, dict)]
    if not pairs:
        return []
    col = colours(before, pairs)
    first = draw(before, col)
    if not first:
        return []

    def card(head: Any, url: str | None, foot: list) -> Any:
        return html.Div([
            html.Div(head, style={"fontSize": "11px", "fontWeight": "700",
                                  "color": "#334155", "margin": "0 0 3px"}),
            (html.Img(src=url, style={"width": "100%", "borderRadius": "6px",
                                      "display": "block"})
             if url else html.Div("picture not found", className="ticket-empty")),
            html.Div(foot, style={"fontSize": "10.5px", "color": "#64748b",
                                  "marginTop": "3px", "lineHeight": "1.5"}),
        ], className="who-card", style={"flex": "0 0 300px", "minWidth": "300px"})

    cards = [card("before", first, [])]
    for t, after in pairs:
        t = t if isinstance(t, dict) else {}
        found = after.get("tracks") if isinstance(after.get("tracks"), dict) else {}
        s = align4.summary(found)
        unsure = set(after.get("unsure") or [])
        # a new object is shown only when the edited run calls it a threat
        # or puts it at risk; a new tree or handrail is left out
        try:
            matter_after = _matter(after)
        except Exception:
            matter_after = set()
        new = [x for x in (found.get("new") or []) if isinstance(x, str) and x in matter_after]
        gone = [x for x in (found.get("gone") or []) if isinstance(x, str) and x in col]
        names = _its_names(after)
        foot: list = [html.Span(says(after), className="who-sure",
                                style={"fontWeight": "700",
                                       "color": "#16a34a" if not s["not sure"] else "#b45309"}),
                      html.Span(f" · same place {s['same place']}"
                                + (f" · by looks {s['by looks']}" if s["by looks"] else ""))]
        if unsure:
            foot.append(html.Div("not sure (a guess, dashed): " + ", ".join(sorted(unsure)),
                                 className="who-unsure", style={"color": "#b45309"}))
        if gone:
            foot.append(html.Div("gone: " + ", ".join(gone), className="who-gone"))
        if new:
            foot.append(html.Div("new: " + ", ".join(new), className="who-new"))
        if names:
            foot.append(html.Div(
                "its own names there: " + ", ".join(f"{h} is {i}" for h, i in sorted(names.items())),
                className="who-names"))
        cards.append(card(f"#{t.get('index', '?')} "
                          + ui_intervene.kind_phrase(t, t.get("target")),
                          draw(after, col, unsure, new, names), foot))
    n = not_sure(pairs)
    return [html.Details([
        html.Summary([html.Span(TITLE, className="iv-title", style=title_style or {}),
                      html.Span(" · " + ("all sure" if not n else f"{n} not sure"),
                                className="who-head",
                                style={"fontWeight": "700", "fontSize": "12px",
                                       "color": "#16a34a" if not n else "#b45309"})],
                     className="iv-fold-head"),
        html.Div("the same object has the same colour in every picture; only the threats and "
                 "those at risk are shown",
                 className="who-key", style={"fontSize": "10.5px", "color": "#94a3b8",
                                             "margin": "2px 0 6px"}),
        html.Div(cards, className="who-strip",
                 style={"display": "flex", "gap": "10px", "overflowX": "auto",
                        "paddingBottom": "6px"}),
    ], className="iv-evidence who-is-who", open=bool(n))]
