"""Clean view of Stage 4 (UI cleanup step 3, agentic/UI_CLEANUP_PLAN.md).

The detailed view is `ui.py` as it was. This file holds the small builders
the clean view puts in front of it: a compact trust card, compact
recommendation cards, one-line gists for the folded sections, and the
footer that names every section the clean view hides. Nothing here reads
a model or a run; every function is a pure function of the Stage 4 record.

Iron rule 8 in UI form: nothing hides silently. A folded section shows a
one-line gist in its header; a hidden section with content is named in
the footer "hidden in clean view: ...".
"""
from __future__ import annotations

from typing import Any, Callable

from dash import html

# What the clean view does with each Stage 4 section:
#   show  — swap the body for the clean builder
#   fold  — keep the detailed body, closed, with a one-line gist
#   hide  — drop it and name it in the footer
#   move  — lives in the FINDINGS card on the left (Sunny, 2026-11-09:
#           the right column is what the model said, the left column is
#           what we found out about it)
CLEAN_POLICY: dict[str, str] = {
    "verdict": "move",
    "recs": "show",
    "stability": "fold",
    "graphs": "fold",
    "alignment": "show",
    "bench": "hide",
    "register": "show",
}

_BAND_COLOUR = {"high": "#16a34a", "moderate": "#b45309", "low": "#b91c1c"}
_WHAT_TO_DO = {
    "high": "rely on it as the model's stable position (not yet proven grounded)",
    "moderate": "usable — review the flagged recommendations before acting",
    "low": "don't rely on it; the model contradicts itself — send to "
           "reflection or a human",
}
_HOLDS_UP = {"advice_backed_by_belief": "every danger it acts on is one it holds",
             "dangers_acted_on": "acts on every danger it sees",
             "uncertainty": "stable on re-ask",
             "internal_alignment": "hangs together",
             "pick_agreement": "targets agree",
             "conformance": "graph well-formed"}
_SMALL = {"fontSize": "11px", "color": "#64748b"}


def _band_colour(band: str) -> str:
    return _BAND_COLOUR.get(str(band or ""), "#64748b")


def _trust_colour(score: float) -> str:
    return "#16a34a" if score >= 0.7 else "#b45309" if score >= 0.4 else "#b91c1c"


def _consequence_colour(band: str) -> str:
    return {"high": "#b91c1c", "medium": "#b45309", "low": "#16a34a"}.get(
        str(band or ""), "#94a3b8")


def _chip(text: str, colour: str, *, fill: bool = False, title: str = "") -> Any:
    style = {"fontSize": "10px", "fontWeight": "800", "borderRadius": "6px",
             "padding": "1px 6px", "marginRight": "6px", "whiteSpace": "nowrap"}
    if fill:
        style.update({"background": colour, "color": "#fff"})
    else:
        style.update({"border": f"1px solid {colour}", "color": colour})
    return html.Span(text, style=style, title=title or None)


# ── 1 · THE VERDICT ──────────────────────────────────────────────────────

def clean_verdict(s4: dict) -> list:
    """Score, band, a bar, what to do, the two biggest pull-downs, and what
    holds up. Minor factors, the per-recommendation list and the footnotes
    stay in the detailed view."""
    trust = (s4 or {}).get("trust") or {}
    score = trust.get("score")
    if score is None:
        return []
    band = str(trust.get("band") or "")
    col = _band_colour(band)
    rows: list[Any] = [
        html.Div([
            html.Span("TRUST ", style={"fontSize": "10px", "fontWeight": "800",
                                       "letterSpacing": ".08em", "color": "#64748b"}),
            html.Span(f"{score} · {band}",
                      style={"fontSize": "16px", "fontWeight": "800", "color": col}),
        ]),
        html.Div(html.Div(style={"width": f"{max(0, min(100, float(score) * 100)):.0f}%",
                                 "height": "6px", "background": col,
                                 "borderRadius": "3px"}),
                 style={"background": "#e2e8f0", "borderRadius": "3px",
                        "margin": "4px 0 6px", "height": "6px"}),
    ]
    wd = _WHAT_TO_DO.get(band)
    if wd:
        rows.append(html.Div([html.Span("what to do: ", style={"fontWeight": "700",
                                                                 "color": "#64748b"}),
                              html.Span(wd, style={"color": col, "fontWeight": "600"})],
                             style={"fontSize": "11.5px", "marginBottom": "4px"}))
    hits = sorted((c for c in (trust.get("contributors") or [])
                   if isinstance(c, dict) and (c.get("contribution") or 0) > 0),
                  key=lambda c: -float(c.get("contribution") or 0))
    if hits:
        label = ("minor notes: " if band == "high" else "pulls down: ")
        rows.append(html.Div(
            [html.Span(label, style={"fontWeight": "700", "color": "#64748b"})]
            + [html.Span(f"{c.get('text', '')} (−{c.get('contribution')})"
                         + (" · " if i < min(2, len(hits)) - 1 else ""),
                         style={"color": "#94a3b8" if band == "high" else col})
               for i, c in enumerate(hits[:2])]
            + ([html.Span(f" · +{len(hits) - 2} more in detailed view",
                          style={"color": "#94a3b8"})] if len(hits) > 2 else []),
            style={"fontSize": "11px", "marginBottom": "2px"}))
    clean = [c for c in (trust.get("contributors") or [])
             if isinstance(c, dict) and (c.get("contribution") or 0) == 0]
    if clean:
        rows.append(html.Div(
            "holds up: " + " · ".join(_HOLDS_UP.get(str(c.get("signal")), str(c.get("signal")))
                                      for c in clean),
            style={"fontSize": "11px", "color": "#16a34a", "fontWeight": "600"}))
    return [html.Div(rows, className="unc-panel",
                     style={"borderColor": col, "padding": "8px 10px"})]


# ── 2 · THE RECOMMENDATIONS ──────────────────────────────────────────────

def _pathology_ranks(s4: dict) -> dict[int, list[str]]:
    """rank -> the pathology kinds whose tickets name that recommendation."""
    out: dict[int, list[str]] = {}
    for tk in ((s4 or {}).get("pathology") or {}).get("tickets") or []:
        if not isinstance(tk, dict):
            continue
        kind = str(tk.get("pathology") or tk.get("kind") or "pathology")
        for r in (tk.get("recs") or []):
            try:
                out.setdefault(int(r), []).append(kind)
            except (TypeError, ValueError):
                continue
    return out


def clean_recs(s4: dict) -> list:
    """One compact card per acting recommendation: rank, action, trust chip,
    consequence chip, a red chip when a pathology ticket names it; the
    reason and the quad fold inside under "why". Benign entries fold under
    one line. Across-recommendations, the picks, the thinking and the
    advisory stay in the detailed view."""
    recs = [r for r in ((s4 or {}).get("recommendations") or []) if isinstance(r, dict)]
    if not recs:
        return []
    per_rec = {str(p.get("rank")): p
               for p in (((s4 or {}).get("trust") or {}).get("per_rec") or [])
               if isinstance(p, dict)}
    flagged = _pathology_ranks(s4)
    cards: list = []
    benign: list = []
    for r in recs:
        rank = r.get("rank")
        q = r.get("structured_reasoning") or {}
        if not isinstance(q, dict):
            q = {}
        if r.get("benign"):
            benign.append(html.Div(f"#{rank} {r.get('action', '')}",
                                   style={"fontSize": "11px", "color": "#475569"}))
            continue
        chips: list = []
        p = per_rec.get(str(rank)) or {}
        if p.get("score") is not None:
            chips.append(_chip(f"trust {p['score']}", _trust_colour(float(p["score"]))))
        if p.get("consequence_band"):
            chips.append(_chip(f"consequence {p['consequence_band']}",
                               _consequence_colour(p["consequence_band"])))
        for kind in dict.fromkeys(flagged.get(int(rank) if isinstance(rank, int) else -1, [])):
            chips.append(_chip(kind.replace("_", " "), "#be123c", fill=True,
                               title="a pathology ticket names this recommendation; "
                                     "see THE REGISTER"))
        quad = (f"{q.get('threat', '')} —{q.get('effect', '')}→ "
                f"{', '.join(q.get('affected_objects') or [])}") if q else ""
        why = [html.Div(str(r.get("reason") or ""), style={"fontSize": "11.5px",
                                                         "color": "#334155"})]
        if quad.strip(" —→"):
            why.append(html.Div(f"quad: {quad}", style={"fontSize": "11px",
                                                       "color": "#7c3aed"}))
        if r.get("expected_consequence"):
            why.append(html.Div(f"expected: {r['expected_consequence']}", style=_SMALL))
        cards.append(html.Div([
            html.Div([html.Span(f"#{rank}", className="phase-num",
                                style={"marginRight": "6px"}),
                      html.Span(str(r.get("action") or ""), style={"fontWeight": "700"})],
                     style={"marginBottom": "3px"}),
            html.Div(chips, style={"marginBottom": "2px"}) if chips else None,
            html.Details([html.Summary("why", style={"fontSize": "10px",
                                                     "color": "#64748b",
                                                     "cursor": "pointer"}),
                          html.Div(why, style={"padding": "2px 0 0 8px"})]),
        ], className="ticket", style={"margin": "6px 0"}))
    if benign:
        n = len(benign)
        cards.append(html.Details([
            html.Summary(f"{n} {'entity' if n == 1 else 'entities'} judged safe "
                         "— no action needed",
                         style={"fontSize": "11px", "color": "#15803d",
                                "cursor": "pointer"}),
            html.Div(benign, style={"padding": "2px 0 0 8px"})],
            style={"margin": "6px 0"}))
    return cards


# ── gists for the folded sections ────────────────────────────────────────

def section_gist(key: str, s4: dict) -> str:
    """One line for a folded section's header, from the record alone."""
    s4 = s4 or {}
    if key == "stability":
        parts = []
        unc = s4.get("uncertainty") or {}
        if unc.get("n_probes"):
            sc = unc.get("score")
            parts.append((f"{sc:.2f} unsure" if isinstance(sc, (int, float)) else f"{sc}")
                         + f" over {unc['n_probes']} re-asks")
        gbu = s4.get("graph_b_uncertainty") or {}
        if gbu.get("n_probes"):
            sc = gbu.get("score")
            parts.append("Graph B " + (f"{sc:.2f} unsure" if isinstance(sc, (int, float))
                                       else f"{sc}"))
        return " · ".join(parts) or "no re-asks on this run"
    if key == "graphs":
        parts = []
        for name, g in (("Graph A", s4.get("graph_a")), ("Graph B", s4.get("graph_b"))):
            if isinstance(g, dict) and (g.get("edges") or g.get("nodes")):
                parts.append(f"{name} {len(g.get('edges') or [])} links")
        conf = s4.get("conformance") or {}
        if conf.get("n_issues") is not None:
            n = conf["n_issues"]
            parts.append("well-formed" if not n else f"{n} rule break{'s' if n != 1 else ''}")
        return " · ".join(parts) or "no graphs on this run"
    if key == "bench":
        n = len(((s4.get("card_judge") or {}).get("verdicts") or []))
        n += 1 if (s4.get("graph_judge") or {}).get("text") else 0
        n += 1 if (s4.get("runoff_judge") or {}) else 0
        return "off this run" if not n else f"{n} card{'s' if n != 1 else ''}"
    return ""


# ── 5 · ALIGNMENT ────────────────────────────────────────────────────────

def clean_alignment(s4: dict) -> list:
    """The verdict sentence, the yardstick warning, and two plain bars.
    Resolved ids, the decomposition and the effect toggle stay detailed."""
    al = (s4 or {}).get("alignment") or {}
    af, bc = al.get("a_fidelity"), al.get("b_coverage")
    if af is None or bc is None:
        return []
    same = af >= 0.8 and bc >= 0.8
    verdict = ("the advice and the model's own causal beliefs tell the same story"
               if same else
               "the advice and the model's own causal beliefs diverge")
    rows: list[Any] = [html.Div(verdict, style={"fontSize": "12px", "fontWeight": "600",
                                                 "color": "#16a34a" if same else "#b45309"})]
    gate = ((s4 or {}).get("trust") or {}).get("graph_b_gate") or {}
    if gate and not gate.get("trusted", True):
        rows.append(html.Div("Graph B failed the yardstick check: these numbers "
                             "are shown but withheld from trust",
                             style={"fontSize": "11px", "color": "#b91c1c",
                                    "fontWeight": "600"}))

    def bar(label: str, value: float, tip: str) -> Any:
        return html.Div([
            html.Div([html.Span(label, style={"fontWeight": "700"}),
                      html.Span(f" {value:.2f}", style={"color": "#64748b"})],
                     style={"fontSize": "11px"}, title=tip),
            html.Div(html.Div(style={"width": f"{max(0, min(100, value * 100)):.0f}%",
                                     "height": "5px", "background": "#7c3aed",
                                     "borderRadius": "3px"}),
                     style={"background": "#e2e8f0", "borderRadius": "3px",
                            "height": "5px", "margin": "2px 0 6px"}),
        ])
    rows.append(bar("advice backed by belief", float(af),
                    "of the causal claims the advice leans on, how many the "
                    "model's own graph backs"))
    rows.append(bar("beliefs acted on", float(bc),
                    "of the claims the model believes, how many the advice acts on"))
    return [html.Div(rows, className="unc-panel", style={"padding": "8px 10px"})]


# ── 7 · THE REGISTER ─────────────────────────────────────────────────────

def _register_of(s4: dict) -> dict:
    reg = (s4 or {}).get("tickets")
    if not reg:
        from agentic.register4 import stage4_register
        reg = stage4_register(s4 or {})
    return reg


def _heading(text: str) -> Any:
    return html.Div(text, style={"fontSize": "10px", "fontWeight": "800",
                                 "letterSpacing": ".08em", "color": "#64748b",
                                 "margin": "8px 0 2px"})


def clean_register(s4: dict, **_ignored: Any) -> list:
    """In the Stage 4 card the register is one count chip for the rule
    tickets and a pointer: the chain and the pathology tickets live in the
    FINDINGS card on the left. The probe answers, the thinking folds and
    "who looked" stay in the detailed view."""
    c = _register_of(s4).get("counts") or {}
    n_rule = int(c.get("rule") or 0)
    n_path = int(c.get("pathology") or 0)
    return [html.Div([
        _chip(f"{n_rule} rule ticket{'s' if n_rule != 1 else ''}",
              "#b45309" if n_rule else "#94a3b8",
              title="code and judge rule violations; the detailed view lists them"),
        _chip(f"{n_path} pathology ticket{'s' if n_path != 1 else ''}",
              "#be123c" if n_path else "#94a3b8", fill=bool(n_path),
              title="in FINDINGS, on the left"),
        html.Span("rule tickets in detailed view · pathology tickets and the "
                  "chain in FINDINGS on the left", style=_SMALL),
    ], style={"margin": "6px 0"})]


def findings(s4: dict, d: dict, *, chain: Callable[[dict], Any],
             pathology_ticket: Callable[[dict], Any]) -> list:
    """The FINDINGS card (clean view, left column, under the scene): what
    we found out about the model's answer. Trust, the run's chain, every
    pathology ticket with its own chain, and the causal-groundedness box
    reserved for the intervention. Before Stage 4 has a result it says
    what it is waiting for."""
    s4 = s4 or {}
    if d.get("stage4_error"):
        return [html.Div(f"stage 4 failed: {d['stage4_error']}",
                         style={"fontSize": "12px", "color": "#b91c1c"})]
    if not s4:
        waiting = ("stage 4 is running — findings arrive with the trust score"
                   if d.get("stage4_started") else
                   "findings arrive once stage 4 has run")
        return [html.Div(waiting, className="ticket-empty")]
    out: list = []
    out += clean_verdict(s4) or [html.Div("trust not scored on this run",
                                          className="ticket-empty")]
    out.append(_heading("THE RUN'S CHAIN"))
    out.append(chain(s4))
    out.append(_heading("PATHOLOGIES"))
    path = _register_of(s4).get("pathology") or []
    out += [pathology_ticket(tk) for tk in path] or [
        html.Div("no pathology ticket", className="ticket-empty")]
    pa = s4.get("pathology") or {}
    looked = pa.get("checked") or []
    if looked:
        out.append(html.Div(f"{len(looked)} detector{'s' if len(looked) != 1 else ''} looked"
                            + (f" · {len(pa.get('not_run') or [])} did not run"
                               if pa.get("not_run") else ""),
                            style={"fontSize": "10.5px", "color": "#94a3b8"}))
    out.append(_heading("CAUSAL GROUNDEDNESS"))
    out.append(html.Div(
        "reserved: suppress the hazard, see whether the plan moves "
        "(the intervention is not yet ported)",
        className="ticket-empty", style={"borderStyle": "dashed"}))
    return out


# ── assembly ─────────────────────────────────────────────────────────────

def hidden_note(hidden: list[str], moved: list[str] | None = None) -> Any | None:
    if not hidden and not moved:
        return None
    parts = []
    if moved:
        parts.append("in FINDINGS on the left: " + " · ".join(moved))
    if hidden:
        parts.append("hidden in clean view: " + " · ".join(hidden)
                     + " — switch to detailed to see them")
    return html.Div(" · ".join(parts),
                    className="only-clean",
                    style={"fontSize": "10.5px", "color": "#94a3b8",
                           "margin": "6px 2px 0"})


def assemble_clean(body: dict[str, list], s4: dict, d: dict, sections: tuple,
                   *, chain: Callable[[dict], Any],
                   pathology_ticket: Callable[[dict], Any]) -> list:
    """Clean-view counterpart of ui._assemble_stage4_sections. `body` is
    the detailed content per section key (already partitioned); `sections`
    is ui._S4_SECTIONS. Every section with content is either shown (clean
    builder, falling back to the detailed body when the builder has
    nothing to say), folded with a gist, or named in the footer."""
    builders: dict[str, Callable[[], list]] = {
        "verdict": lambda: clean_verdict(s4),
        "recs": lambda: clean_recs(s4),
        "alignment": lambda: clean_alignment(s4),
        "register": lambda: clean_register(s4, chain=chain,
                                           pathology_ticket=pathology_ticket),
    }
    out: list = []
    hidden: list[str] = []
    moved: list[str] = []
    for key, title, sub, open_ in sections:
        if title is None or not body.get(key):
            continue
        policy = CLEAN_POLICY.get(key, "fold")
        if policy == "move":
            moved.append(title.split("·", 1)[-1].strip().lower())
            continue
        if policy == "hide":
            gist = section_gist(key, s4)
            hidden.append(title.split("·", 1)[-1].strip().lower()
                          + (f" ({gist})" if gist else ""))
            continue
        if policy == "show":
            content = builders[key]() if key in builders else []
            content = content or body[key]
            head_sub = sub
            is_open = True
        else:
            content = body[key]
            head_sub = section_gist(key, s4) or sub
            is_open = False
        out.append(html.Details([
            html.Summary([
                html.Span(title, style={"fontSize": "12px", "fontWeight": "800",
                                        "letterSpacing": ".04em"}),
                html.Span("  — " + head_sub, style={"fontSize": "10.5px",
                                                    "color": "#94a3b8"}),
            ], style={"cursor": "pointer", "padding": "6px 8px",
                      "background": "#f1f5f9", "borderRadius": "6px"}),
            html.Div(content, style={"padding": "6px 0 2px 4px"}),
        ], open=is_open, style={"marginBottom": "8px"}))
    note = hidden_note(hidden, moved)
    if note is not None:
        out.append(note)
    return out
