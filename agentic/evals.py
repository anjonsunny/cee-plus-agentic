"""The eval battery for Arm B, wired STAGE-WISE plus one global roll-up.

WHY STAGE-WISE (Sunny 2026-07-22)
=================================
One end-score cannot say WHERE an error was born. Each stage is scored
against its own GT, and stage-2 GT is CONDITIONED on the frozen stage-1
output — so a stage-2 miss that traces to a stage-1 GT miss is charged
to perception, not to assessment. That attribution is also the measured
justification for re-perception: when eval shows stage-2 errors caused
by stage-1 misses (B_pool: the model keeps inventing a 'pool' threat
because perception never declared the water hazard), a petition upstream
is evidence-backed, not a hunch.

THE FOUR TECHNIQUES (all here)
==============================
1. GT harness + the REFLECTION QUADRANT: pre- vs post-reflection answers
   scored against GT; combined with U0->U1:
                        dU down          dU up
     toward GT       REFINEMENT       NOISY IMPROVEMENT
     away from GT    FALSE CERTAINTY  DESTABILIZATION
2. Talking-points counts (objective, judge-free): do reasons cite the
   entity's declared state? does reasoning name specific entities?
3. LLM-as-judge, blind pairwise: pre vs post shown unlabeled in
   deterministic-shuffled order; judge (DIFFERENT training family than
   the subject — plan §10) picks which better fits the declared states.
4. Rubric judging: per-reason 0/1 rubric, and STOOD-justification
   scoring (standing ground is allowed but hedging is labeled).

GT PROTOCOL: Claude proposed experiments/agentic_scenes/gt_stage2.json,
Sunny verifies (cross-model reference, not strict truth). Eval NEVER
feeds the pipeline — the instrument must not grade itself into the
right answer; these numbers are for us.

Run:  python -m agentic.evals            # stage-wise + quadrant tables
      python -m agentic.evals --judge    # adds judge passes (Ollama)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.uncertainty import type_family  # noqa: E402

GT_PATH = REPO_ROOT / "experiments" / "agentic_scenes" / "gt_stage2.json"
PERCEPTION_DIR = REPO_ROOT / "experiments" / "agentic_scenes" / "perception"
ASSESSMENT_DIR = REPO_ROOT / "experiments" / "agentic_scenes" / "assessment"

BUCKET_ORDER = ["none", "minor", "serious", "catastrophic"]


def load_gt(path: Path = GT_PATH) -> dict[str, Any]:
    gt = json.loads(path.read_text())
    gt.pop("_comment", None)
    return gt


# ── Stage 1 eval: did perception declare the key entities? ──────────────


def _matches(obj: dict[str, Any], want: dict[str, Any]) -> bool:
    if obj.get("label") != want.get("label"):
        return False
    if "state" in want and obj.get("state") != want["state"]:
        return False
    if "state_kind" in want and obj.get("state_kind") != want["state_kind"]:
        return False
    return True


def eval_stage1(record: dict[str, Any], gt1: dict[str, Any]) -> dict[str, Any]:
    """Label-based recall of the scene-truth key entities, plus the
    hazard question. A required entity nobody declared is a PERCEPTION
    miss — the charge sheet re-perception petitions will cite."""
    objs = [o if isinstance(o, dict) else o.model_dump()
            for o in record.get("detected_objects", [])]
    misses, hits = [], []
    used: set[int] = set()
    for want in gt1.get("required", []):
        found = next((i for i, o in enumerate(objs)
                      if i not in used and _matches(o, want)), None)
        if found is None:
            misses.append({k: v for k, v in want.items()
                           if not k.startswith("_")})
        else:
            used.add(found)
            hits.append(objs[found]["object_id"])
    has_hazard = any(o.get("state_kind") == "hazard_bearing" for o in objs)
    n_req = len(gt1.get("required", []))
    return {
        "required_recall": round(len(hits) / n_req, 3) if n_req else 1.0,
        "missed_entities": misses,
        "hazard_expected": gt1.get("expect_hazard"),
        "hazard_declared": has_hazard,
        "false_hazard": has_hazard and not gt1.get("expect_hazard", True),
        "missing_hazard": (not has_hazard) and gt1.get("expect_hazard", False),
    }


# ── Stage 2 eval: the conditioned assessment score ──────────────────────


def eval_stage2(a: dict[str, Any], gt2: dict[str, Any]) -> dict[str, Any]:
    """Score one assessment dict against the conditioned GT. Returns the
    component judgments plus a single error_score (lower = better) used
    by the quadrant: verdict miss is the cardinal sin (weight 3), bucket
    distance 1/step, missed threat or at-risk 1, spurious 1, wrong kind
    0.5, wrong type family 1."""
    out: dict[str, Any] = {}
    out["verdict_ok"] = a.get("disaster_scenario") == gt2["disaster_scenario"]
    fam = type_family(a.get("disaster_type", ""))
    out["type_family_ok"] = (fam in gt2["type_families"]
                             or gt2["disaster_scenario"] == "No" and fam == "none")
    bucket = a.get("severity_bucket", "none")
    out["bucket_ok"] = bucket in gt2.get("bucket_acceptable", [gt2["bucket"]])
    try:
        out["bucket_distance"] = abs(BUCKET_ORDER.index(bucket)
                                     - BUCKET_ORDER.index(gt2["bucket"]))
    except ValueError:
        out["bucket_distance"] = len(BUCKET_ORDER)

    got_threats = {t.get("object_id") for t in a.get("threats", [])}
    want_threats = set(gt2.get("threats", []))
    ok_threats = want_threats | set(gt2.get("threats_optional", []))
    out["threat_recall"] = (round(len(got_threats & want_threats)
                                  / len(want_threats), 3)
                            if want_threats else 1.0)
    out["threat_spurious"] = sorted(got_threats - ok_threats)

    got_risk = {r.get("object_id"): r.get("kind")
                for r in a.get("at_risk", [])}
    want_risk = gt2.get("at_risk", {})
    ok_risk = {**want_risk, **gt2.get("at_risk_optional", {})}
    out["at_risk_recall"] = (round(len(set(got_risk) & set(want_risk))
                                   / len(want_risk), 3)
                             if want_risk else 1.0)
    out["at_risk_spurious"] = sorted(set(got_risk) - set(ok_risk))
    out["kind_errors"] = sorted(
        oid for oid, kind in got_risk.items()
        if oid in ok_risk and kind != ok_risk[oid])

    out["error_score"] = round(
        (0 if out["verdict_ok"] else 3)
        + (0 if out["type_family_ok"] else 1)
        + out["bucket_distance"]
        + (len(want_threats) - len(got_threats & want_threats))
        + len(out["threat_spurious"])
        + (len(want_risk) - len(set(got_risk) & set(want_risk)))
        + len(out["at_risk_spurious"])
        + 0.5 * len(out["kind_errors"]), 2)
    return out


# ── The reflection quadrant ─────────────────────────────────────────────


def quadrant(err_pre: float, err_post: float,
             u_before: Optional[float],
             u_after: Optional[float]) -> str:
    """Classify one scene's reflection outcome. ΔU alone proves nothing;
    ΔU x Δ(GT error) is the measurement."""
    if err_post == err_pre and (u_after is None or u_after == u_before):
        return "unchanged"
    toward = err_post < err_pre
    away = err_post > err_pre
    du_down = (u_before is not None and u_after is not None
               and u_after < u_before)
    du_up = (u_before is not None and u_after is not None
             and u_after > u_before)
    if toward:
        return "REFINEMENT" if du_down else \
            ("noisy improvement" if du_up else "improvement (U flat)")
    if away:
        return "FALSE CERTAINTY" if du_down else \
            ("DESTABILIZATION" if du_up else "regression (U flat)")
    return "U moved, answer GT-equal"


# ── Talking-points counts (objective, judge-free) ───────────────────────


def citation_counts(a: dict[str, Any],
                    record: dict[str, Any]) -> dict[str, Any]:
    """Cheap groundedness proxies over the free text: does each cited
    entity's reason mention its DECLARED state word? does the overall
    reasoning name specific entities?"""
    states = {o["object_id"]: str(o.get("state", ""))
              for o in record.get("detected_objects", [])}
    entries = list(a.get("threats", [])) + list(a.get("at_risk", []))
    cited = 0
    uncited: list[str] = []
    for e in entries:
        oid = e.get("object_id")
        state = states.get(oid, "")
        text = str(e.get("reason", "")).lower()
        root = state.rstrip("ing").rstrip("ed")     # burning->burn, damaged->damag
        if state and root and root.lower() in text:
            cited += 1
        else:
            uncited.append(str(oid))
    reasoning = str(a.get("reasoning", ""))
    named = [oid for oid in states if oid in reasoning]
    # S8 surface (2026-07-22): threat reasons that describe RECEIVING harm
    # — the code check triggers reflection; this line is the visibility.
    from agentic.assessment import victim_shaped
    victim = [str(t.get("object_id")) for t in a.get("threats", [])
              if victim_shaped(t.get("reason", ""))]
    return {
        "n_entries": len(entries),
        "state_cited": cited,
        "state_citation_rate": (round(cited / len(entries), 3)
                                if entries else None),
        "reasoning_names_entities": len(named),
        "uncited": uncited,
        "victim_shaped": victim,
    }


# ── LLM-as-judge (different training family; injectable for tests) ─────

JudgeFn = Callable[[str], str]


def _ollama_judge(prompt: str) -> str:
    """The judge model. DIFFERENT training family than the subject VLM
    (plan §10: same lineage shares the tendencies the judge must catch).
    Default llama3.1:8b — pull with `ollama pull llama3.1:8b`."""
    import requests

    api_url = os.getenv("QWEN_API_URL",
                        "http://localhost:11434/v1/chat/completions")
    r = requests.post(api_url, json={
        "model": os.getenv("JUDGE_MODEL", "llama3.1:8b"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }, timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def _states_block(record: dict[str, Any]) -> str:
    return "\n".join(f"  - {o['object_id']}: {o.get('label')} · state="
                     f"{o.get('state')} ({o.get('state_kind')})"
                     for o in record.get("detected_objects", []))


def _evidence_block(record: dict[str, Any]) -> str:
    """EVERYTHING the assessor was allowed to see, so the judge judges on
    the same evidence basis: caption + declared states + geometry hints.
    Never the image (text-only discipline extends to judges)."""
    from agentic.geometry import hints_as_prompt_lines, spatial_hints
    hints = spatial_hints(record.get("detected_objects", []),
                          record.get("image_size"))
    return (f"CAPTION: {record.get('caption') or '(none)'}\n"
            f"DECLARED STATES:\n{_states_block(record)}\n"
            f"SPATIAL HINTS (2D nominations from declared boxes):\n"
            f"{hints_as_prompt_lines(hints)}")


def substantive_key(a: dict[str, Any]) -> tuple:
    """The DECISION layer of an assessment — the fields downstream stages
    consume — folded the same way the probe machinery folds them:
    scenario, type FAMILY (wording ignored), severity BUCKET (raw level
    ignored), threat set, at-risk set with kinds. Two answers with equal
    keys differ only in narration; a pairwise judge asked to rank them is
    answering an undefined question (C_tanker ui_529ce417: llama picked
    POST between two identical decisions and confabulated a reason)."""
    from agentic.assessment import severity_bucket
    from agentic.uncertainty import type_family
    return (
        str(a.get("disaster_scenario")),
        type_family(a.get("disaster_type", "")),
        severity_bucket(int(a.get("disaster_level") or 0)),
        frozenset(str(t.get("object_id")) for t in a.get("threats") or []),
        frozenset((str(r.get("object_id")), str(r.get("kind")))
                  for r in a.get("at_risk") or []),
    )


def judge_pairwise(pre: dict[str, Any], post: dict[str, Any],
                   record: dict[str, Any], scene: str,
                   judge_fn: JudgeFn | None = None) -> dict[str, Any]:
    """Blind pairwise: pre and post shown as A/B in an order derived
    deterministically from the scene name (reproducible, not gameable by
    position bias in one fixed direction). Returns winner: pre|post|tie."""
    judge_fn = judge_fn or _ollama_judge
    flipped = random.Random(scene).random() < 0.5
    first, second = (post, pre) if flipped else (pre, post)
    prompt = (
        "Two assessments of the same emergency scene are shown. Judge "
        "ONLY against the declared evidence below — not against "
        "what you imagine the scene contains.\n\n"
        f"{_evidence_block(record)}\n\n"
        f"ASSESSMENT A:\n{json.dumps(first, indent=1)}\n\n"
        f"ASSESSMENT B:\n{json.dumps(second, indent=1)}\n\n"
        "Which assessment better fits the declared states? Answer with "
        "exactly one word first — A, B, or TIE — then one sentence why.")
    answer = judge_fn(prompt)
    token = answer.strip().split()[0].strip(".,:").upper() if answer.strip() else "TIE"
    if token not in ("A", "B", "TIE"):
        return {"winner": "unparseable", "raw": answer[:200]}
    if token == "TIE":
        winner = "tie"
    else:
        # flipped -> post was shown as A; not flipped -> pre was A
        winner = "post" if ((token == "A") == flipped) else "pre"
    return {"winner": winner, "raw": answer[:200]}


def judge_runoff(top1: dict[str, Any], top2: dict[str, Any],
                 record: dict[str, Any], scene: str,
                 judge_fn: JudgeFn | None = None) -> dict[str, Any]:
    """Blind runoff between the two most-voted probe candidates (high-U
    case: the model's own readings disagree; an independent judge says
    which fits the declared states). Winner: top1|top2|tie. The result
    is CONTEXT for reflection — it is never installed as the answer."""
    judge_fn = judge_fn or _ollama_judge
    flipped = random.Random("runoff:" + scene).random() < 0.5
    first, second = (top2, top1) if flipped else (top1, top2)
    prompt = (
        "Two candidate assessments of the same emergency scene are shown "
        "(the same model produced both under resampling). Judge ONLY "
        "against the declared evidence below.\n\n"
        f"{_evidence_block(record)}\n\n"
        f"CANDIDATE A:\n{json.dumps(first, indent=1)}\n\n"
        f"CANDIDATE B:\n{json.dumps(second, indent=1)}\n\n"
        "Which candidate better fits the declared states? Answer with "
        "exactly one word first — A, B, or TIE — then one sentence citing "
        "the specific states that decide it.")
    answer = judge_fn(prompt)
    token = (answer.strip().split()[0].strip(".,:").upper()
             if answer.strip() else "TIE")
    if token not in ("A", "B", "TIE"):
        return {"winner": "unparseable", "raw": answer[:200]}
    winner = ("tie" if token == "TIE"
              else "top2" if ((token == "A") == flipped) else "top1")
    return {"winner": winner, "raw": answer[:240]}


RUBRIC_PROMPT = """Score this {what} against the declared states, with a
strict 0/1 rubric. Declared states:
{states}

{what_upper}: {text}
Entity in question: {oid} (declared state: {state})

Rubric (answer each with 0 or 1, then a final line "SCORE: n/2"):
R1. Does the text cite the entity's DECLARED state (not an imagined one)?
R2. Is the causal claim specific (names what harms what), not generic
    filler ("poses a significant risk")?"""


def rubric_reasons(a: dict[str, Any], record: dict[str, Any],
                   judge_fn: JudgeFn | None = None) -> list[dict[str, Any]]:
    """Per-entry reason rubric (technique 4). Returns one row per cited
    entity with the judge's R1/R2 verdicts parsed out."""
    judge_fn = judge_fn or _ollama_judge
    states = {o["object_id"]: o for o in record.get("detected_objects", [])}
    rows = []
    for what, entries in (("threat reason", a.get("threats", [])),
                          ("at-risk reason", a.get("at_risk", []))):
        for e in entries:
            oid = e.get("object_id", "?")
            o = states.get(oid, {})
            answer = judge_fn(RUBRIC_PROMPT.format(
                what=what, what_upper=what.upper(),
                states=_evidence_block(record),
                text=e.get("reason", "(no reason given)"),
                oid=oid, state=o.get("state", "unknown")))
            score = None
            for line in answer.splitlines():
                if "SCORE:" in line.upper():
                    try:
                        score = int(line.upper().split("SCORE:")[1]
                                    .strip().split("/")[0])
                    except (ValueError, IndexError):
                        pass
            rows.append({"entry": what, "object_id": oid,
                         "score": score, "raw": answer[:200]})
    return rows


# ── The harness: score every frozen scene, stage-wise + global ──────────


def _pre_assessment(rec: dict[str, Any],
                    perception: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the PRE-reflection answer: raw_answer is the canonical
    model output verbatim; re-parse and re-derive kinds exactly as the
    pipeline did, on a fresh object (never mutating the stored one)."""
    from agentic.assessment import enforce_kinds, parse_assessment
    from agentic.perception import PerceptionResult
    a, _notes = parse_assessment(rec.get("raw_answer"))
    enforce_kinds(a, PerceptionResult.model_validate(perception))
    return a.model_dump()


def evaluate_scene(name: str, gt: dict[str, Any],
                   perception: dict[str, Any],
                   assessment_rec: dict[str, Any]) -> dict[str, Any]:
    s1 = eval_stage1(perception, gt[name]["stage1"])
    post = assessment_rec["assessment"]
    pre = _pre_assessment(assessment_rec, perception)
    e_pre = eval_stage2(pre, gt[name]["stage2"])
    e_post = eval_stage2(post, gt[name]["stage2"])
    tr = assessment_rec.get("reflection_trace") or {}
    return {
        "scene": name,
        "stage1": s1,
        "pre": e_pre, "post": e_post,
        "quadrant": quadrant(e_pre["error_score"], e_post["error_score"],
                             tr.get("u_before"), tr.get("u_after")),
        "citations": citation_counts(post, perception),
        "attribution": ("stage1 (perception miss shapes stage2 errors)"
                        if s1["missed_entities"] and not e_post["verdict_ok"]
                        or s1["missing_hazard"] else "stage2"),
        "_pre_assessment": pre,
    }


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    use_judge = "--judge" in args
    gt = load_gt()
    rows = []
    for name in sorted(gt):
        p_path = PERCEPTION_DIR / f"{name}__perception.json"
        a_path = ASSESSMENT_DIR / f"{name}__assessment.json"
        if not (p_path.exists() and a_path.exists()):
            print(f"-- {name}: missing records, skipped")
            continue
        perception = json.loads(p_path.read_text())
        assessment_rec = json.loads(a_path.read_text())
        r = evaluate_scene(name, gt, perception, assessment_rec)
        rows.append((r, perception, assessment_rec))

    print(f"{'scene':<16}{'S1 recall':<11}{'pre err':<9}{'post err':<10}"
          f"{'U0->U1':<14}{'quadrant':<22}{'cite%':<7}{'attribution'}")
    for r, _p, rec in rows:
        tr = rec.get("reflection_trace") or {}
        u = (f"{tr.get('u_before')}->{tr.get('u_after')}"
             if tr.get("u_before") is not None else "-")
        cite = r["citations"]["state_citation_rate"]
        print(f"{r['scene']:<16}{r['stage1']['required_recall']:<11}"
              f"{r['pre']['error_score']:<9}{r['post']['error_score']:<10}"
              f"{u:<14}{r['quadrant']:<22}"
              f"{('-' if cite is None else cite):<7}{r['attribution']}")
        for m in r["stage1"]["missed_entities"]:
            print(f"    S1 MISS: {m}  <- re-perception charge sheet")
        for k in ("threat_spurious", "at_risk_spurious", "kind_errors"):
            if r["post"][k]:
                print(f"    post {k}: {r['post'][k]}")

    total_pre = sum(r["pre"]["error_score"] for r, _, _ in rows)
    total_post = sum(r["post"]["error_score"] for r, _, _ in rows)
    print(f"\nGLOBAL: pre-reflection error {total_pre} -> post "
          f"{total_post} across {len(rows)} scenes")

    if use_judge:
        print("\nJUDGE (blind pairwise, pre vs post):")
        for r, perception, rec in rows:
            v = judge_pairwise(r["_pre_assessment"],
                               rec["assessment"], perception, r["scene"])
            print(f"  {r['scene']:<16} winner: {v['winner']}  {v.get('raw', '')[:90]}")


if __name__ == "__main__":
    from agentic.evals import main as _canonical_main
    _canonical_main()
