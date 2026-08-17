"""Stage 4 — the RUNOFF judge (JUDGES.md §4). Advisory and display-only.

WHAT IT IS
==========
The subject answers the same question 5 times at raised temperature (the
probes). When those answers differ, this judge is shown TWO of them — the two
leading candidates — and asked which one the RECORD supports. It runs twice:

    on the recommendation candidates   which advice covers the scene?
    on the Graph B candidates          which causal account is supported?

Two SEPARATE applications of one module, per Sunny: "runoff judge can be used
for all uncertainty candidates... whether it's graph B or recs. Separately."

THE TWINS (JUDGES.md §4, decided 2026-08-08)
============================================
Every verdict is produced twice by the SAME judge model:

    text-only      sees the record, never the image. THE OFFICIAL VERDICT —
                   the only one that will ever feed reflection or a capture
                   label.
    image-aware    same prompt + the image as context, prefixed with the
                   constraint that the entity list IS what was extracted from
                   the image. The second witness.

Only the agreement is shown (`twins_agree`); no routing, no arbitration —
what to do with disagreement is deliberately undecided until live runs show
how the twins actually split. One model in both seats means the image is the
ONLY variable between them.

WHAT THE JUDGE IS NOT ALLOWED TO DO
===================================
Prefer an answer because it likes it. The prompt defines COVERS / SUPPORTED
in checkable clauses (the F28 lesson: the card judge went 0-for-5 through
four rewrites until one missing definition was added), and instructs the
judge to verify clauses, not taste. `equally_good` is a legal verdict —
forcing a winner between equivalent answers manufactures preference noise,
and this judge family has a recorded severity-minimizing lean (F4).

The judge is also never told the vote counts. Tell it one answer got 4 of 5
votes and it anchors on the majority instead of reading. The majority note is
added later, at carry time, when a verdict enters a reflection prompt — the
F5 guard belongs there, not here.

CAPTURE
=======
Every judged pair is a §9.2 preference record in waiting: both candidates
ride VERBATIM in the emitted event, with both twins' verdicts, the vote
splits, the judges' reasoning texts, the prompt version, and the code-decided
facts (invented ids per side). `verified_by_intervention` stays null until
the S6 join fills it.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.judge_card import (JUDGE_PROBE_TEMPERATURE, _ollama_judge,
                                _quad_text, _vote, read_verdict)

# Stamped into every emitted verdict. Bump on ANY prompt edit: pairs judged by
# different prompt versions must be separable forever (JUDGES.md §9.8 — data
# from a later-fixed judge has to be findable to drop).
PROMPT_VERSION = "runoff-v1"

ANSWER_A, ANSWER_B, EQUAL = "answer_a", "answer_b", "equally_good"
_ALLOWED = (ANSWER_A, ANSWER_B, EQUAL)

# The image-aware twin's framing (Sunny's wording, JUDGES.md §4): the record
# is presented AS the extraction from the image, and the image is context.
IMAGE_PREFIX = """The entity list below is what was extracted from the image
you are shown — restrict yourself to these detected objects and their
relationships; use the image as context.

"""

# Prompt-neutral: no scene names, no object ids in the template. The scene and
# both candidates arrive as injected data blocks.
RECS_RUNOFF_PROMPT = """THE SCENE (entities and their states):
{scene_block}

DECLARED IN THE ASSESSMENT:
  dangers:  {threats}
  at risk:  {at_risk}

A model was asked for ranked emergency recommendations for this scene,
{n_asks} separate times. The two leading candidate answers are below,
complete and unedited.

ANSWER A:
{candidate_a}

ANSWER B:
{candidate_b}

An answer COVERS this scene when all three hold:
  1. every declared danger is acted on by at least one of its actions;
  2. every entity declared at risk is protected by at least one of its
     actions — acting on a danger and protecting someone at risk are both
     legitimate ways to cover something;
  3. every entity id the answer mentions appears in the scene's entity list
     above.

Which answer covers this scene with fewer failures? Judge only against the
scene and the declarations — never against what you yourself would
recommend. If both answers cover the scene equally well, say so.

Think it through step by step, citing entity ids as you go. Then end with
exactly this line and nothing after it, choosing one option:

VERDICT: [answer_a | answer_b | equally_good]
"""

GRAPH_RUNOFF_PROMPT = """THE SCENE (entities and their states):
{scene_block}

A model was asked which entities endanger which in this scene, {n_asks}
separate times. The two leading candidate answers are below, complete and
unedited.

ANSWER A:
{candidate_a}

ANSWER B:
{candidate_b}

An answer is SUPPORTED by this scene when all three hold:
  1. every entity id it names appears in the scene's entity list above;
  2. every arrow runs FROM an entity whose state makes it dangerous;
  3. every entity whose state makes it dangerous has at least one outgoing
     arrow.

Which answer is supported with fewer failures? Judge only against the scene
above — never against what you yourself believe endangers what. If both are
supported equally well, say so.

Think it through step by step, citing entity ids as you go. Then end with
exactly this line and nothing after it, choosing one option:

VERDICT: [answer_a | answer_b | equally_good]
"""


# ── Candidate selection ─────────────────────────────────────────────────


def _rec_key(recs: list) -> tuple:
    """What makes two probe answers 'the same advice': the quads, order-blind.
    Wording is NOT identity — a reworded same-action answer is one candidate,
    which is the same wording-invariance rule compute_shifts uses."""
    out = []
    for r in (recs or []):
        if not isinstance(r, dict):
            continue
        q = r.get("structured_reasoning") or {}
        out.append((str(q.get("threat", "")), str(q.get("effect", "")),
                    tuple(sorted(str(a) for a in (q.get("affected_objects")
                                                  or [])))))
    return tuple(sorted(out))


def _graph_key(graph: dict) -> tuple:
    return tuple(sorted((str(e.get("source", "")), str(e.get("effect", "")),
                         str(e.get("target", "")))
                        for e in ((graph or {}).get("edges") or [])
                        if isinstance(e, dict)))


def pick_pair(items: list, key_fn: Any) -> tuple[int, int] | None:
    """The two probes to judge. Groups identical answers, then:

    - fewer than two distinct answers → None. Unanimity needs no runoff, and
      judging two copies of the same answer measures nothing.
    - clear top-2 by votes → their exemplars.
    - tie (D_aerial live: five answers, one vote each) → the two MOST
      DIFFERENT answers, by symmetric difference of their keys. One
      comparison should carry the maximum information.
    """
    groups: dict[tuple, list[int]] = {}
    for i, it in enumerate(items or []):
        groups.setdefault(key_fn(it), []).append(i)
    if len(groups) < 2:
        return None
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
    top_votes = len(ranked[0][1])
    leaders = [kv for kv in ranked if len(kv[1]) == top_votes]
    if len(leaders) >= 2:
        # tie — pick the most different pair among the tied leaders
        best, pair = -1, (leaders[0][1][0], leaders[1][1][0])
        for i in range(len(leaders)):
            for j in range(i + 1, len(leaders)):
                a, b = set(leaders[i][0]), set(leaders[j][0])
                d = len(a ^ b)
                if d > best:
                    best = d
                    pair = (leaders[i][1][0], leaders[j][1][0])
        return pair
    return (ranked[0][1][0], ranked[1][1][0])


# ── Rendering a candidate for the prompt ────────────────────────────────


def _fmt_recs(recs: list) -> str:
    lines = []
    for r in (recs or []):
        if not isinstance(r, dict):
            continue
        q = r.get("structured_reasoning") or {}
        lines.append(f"  {r.get('rank', '?')}. action: {r.get('action', '')}")
        lines.append(f"     reason: {r.get('reason', '')}")
        lines.append(f"     causal claim: {_quad_text(q)}")
    return "\n".join(lines) or "  (no recommendations)"


def _fmt_graph(graph: dict) -> str:
    lines = [f"  {e.get('source')} --{e.get('effect')}--> {e.get('target')}"
             for e in ((graph or {}).get("edges") or []) if isinstance(e, dict)]
    return "\n".join(lines) or "  (no causal links)"


# ── Code facts: the audit rail on every pair (JUDGES.md §9.2) ───────────


def _invented_ids(text_ids: set, record: Any) -> list[str]:
    known = {str(getattr(o, "object_id", "")) for o in
             (getattr(record, "detected_objects", None) or [])}
    return sorted(i for i in text_ids if i and i not in known)


def _ids_in_recs(recs: list) -> set:
    out: set = set()
    for r in (recs or []):
        if not isinstance(r, dict):
            continue
        q = r.get("structured_reasoning") or {}
        if q.get("threat"):
            out.add(str(q["threat"]))
        out |= {str(a) for a in (q.get("affected_objects") or [])}
    return out


def _ids_in_graph(graph: dict) -> set:
    out: set = set()
    for e in ((graph or {}).get("edges") or []):
        if isinstance(e, dict):
            out |= {str(e.get("source", "")), str(e.get("target", ""))}
    return {i for i in out if i}


# ── The judge itself, twice ─────────────────────────────────────────────


def _ollama_judge_image(prompt: str, image_b64: str,
                        temperature: float = 0.0) -> str:
    """The image-aware twin's transport: same judge model, same endpoint, the
    image riding beside the prompt. One model in both seats — the image is the
    only variable between the twins."""
    import requests

    from agentic import models as _models
    r = requests.post(os.getenv("QWEN_API_URL",
                                "http://localhost:11434/v1/chat/completions"),
                      json={"model": _models.JUDGE_VISION_MODEL,
                            "messages": [{"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                           + image_b64}}]}],
                            "temperature": temperature},
                      timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def _probe_twin(prompt: str, judge_fn: Any, n_probes: int) -> dict[str, Any]:
    readings, errors, texts = [], [], []
    for _ in range(max(1, n_probes)):
        try:
            answer = judge_fn(prompt)
        except Exception as exc:            # an unreachable judge must never
            errors.append(str(exc)[:80])    # take the run down
            answer = ""
        texts.append(str(answer))
        readings.append(read_verdict(answer, _ALLOWED))
    out = _vote(readings)
    # capture §9.3: the judge's REASONING, not just its verdict. One exemplar
    # per twin — the answer that voted with the majority, verbatim.
    keep = next((t for t, r in zip(texts, readings)
                 if r == out["verdict"]), texts[0] if texts else "")
    out["reasoning"] = keep[:4000]
    if errors:
        out["errors"] = errors[:3]
    return out


def runoff(application: str, prompt: str, cand_a: str, cand_b: str,
           code_facts: dict, *, judge_fn: Any, judge_image_fn: Any = None,
           n_probes: int = 5) -> dict[str, Any]:
    """One runoff: text twin always, image twin when transport for it exists.
    `twins_agree` is None when the image twin did not run — unknown is not
    agreement."""
    text = _probe_twin(prompt, judge_fn, n_probes)
    image = None
    if judge_image_fn is not None:
        image = _probe_twin(IMAGE_PREFIX + prompt, judge_image_fn, n_probes)
    return {
        "application": application, "advisory": True,
        "prompt_version": PROMPT_VERSION, "n_probes": n_probes,
        "candidate_a": cand_a, "candidate_b": cand_b,
        "text": text, "image": image,
        "twins_agree": (None if image is None
                        else text["verdict"] == image["verdict"]),
        "code_facts": code_facts,
        "verified_by_intervention": None,
    }


def runoff_recommendations(probe_recs: list, record: Any, assessment: Any, *,
                           judge_fn: Any, judge_image_fn: Any = None,
                           n_probes: int = 5) -> dict[str, Any]:
    """The runoff over the recommendation probes (full prose — F49 data)."""
    pair = pick_pair(probe_recs, _rec_key)
    if pair is None:
        return {}
    i, j = pair
    from agentic.recommend import _scene_block
    threats = ", ".join(t.object_id for t in
                        (getattr(assessment, "threats", None) or [])) or "(none)"
    at_risk = ", ".join(a.object_id for a in
                        (getattr(assessment, "at_risk", None) or [])) or "(none)"
    prompt = RECS_RUNOFF_PROMPT.format(
        scene_block=_scene_block(record, assessment), threats=threats,
        at_risk=at_risk, n_asks=len(probe_recs),
        candidate_a=_fmt_recs(probe_recs[i]),
        candidate_b=_fmt_recs(probe_recs[j]))
    facts = {"probe_a": i, "probe_b": j,
             "invented_ids_a": _invented_ids(_ids_in_recs(probe_recs[i]), record),
             "invented_ids_b": _invented_ids(_ids_in_recs(probe_recs[j]), record)}
    return runoff("recommendations", prompt, _fmt_recs(probe_recs[i]),
                  _fmt_recs(probe_recs[j]), facts, judge_fn=judge_fn,
                  judge_image_fn=judge_image_fn, n_probes=n_probes)


def runoff_graph_b(probe_graphs: list, record: Any, assessment: Any, *,
                   judge_fn: Any, judge_image_fn: Any = None,
                   n_probes: int = 5) -> dict[str, Any]:
    """The runoff over the Graph B probes (stored whole since day one)."""
    pair = pick_pair(probe_graphs, _graph_key)
    if pair is None:
        return {}
    i, j = pair
    from agentic.recommend import _scene_block
    prompt = GRAPH_RUNOFF_PROMPT.format(
        scene_block=_scene_block(record, assessment),
        n_asks=len(probe_graphs),
        candidate_a=_fmt_graph(probe_graphs[i]),
        candidate_b=_fmt_graph(probe_graphs[j]))
    facts = {"probe_a": i, "probe_b": j,
             "invented_ids_a": _invented_ids(_ids_in_graph(probe_graphs[i]), record),
             "invented_ids_b": _invented_ids(_ids_in_graph(probe_graphs[j]), record)}
    return runoff("graph_b", prompt, _fmt_graph(probe_graphs[i]),
                  _fmt_graph(probe_graphs[j]), facts, judge_fn=judge_fn,
                  judge_image_fn=judge_image_fn, n_probes=n_probes)


def default_image_judge(image_path: str) -> Any | None:
    """Build the image twin's judge_fn from a scene image on disk, or None
    when there is no image to show."""
    p = Path(image_path or "")
    if not p.is_file():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode()
    return lambda prompt: _ollama_judge_image(
        prompt, b64, temperature=JUDGE_PROBE_TEMPERATURE)
