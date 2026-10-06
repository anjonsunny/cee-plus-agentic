"""Stage 4's ticket register — one place every finding lands, code and judge
alike, in the lifecycle grammar Stage 1 and Stage 2 already use.

Sunny (2026-09-21): "we also don't show the tickets for judges. Judges who
will inform in reflection." Until now a judge's verdict lived only on its
bench card; a code finding lived only in the panel that found it. Reflection
(build step 4) has to carry both, so both become tickets NOW, in one
register, and reflection will read the register instead of re-collecting.

Two registers, per JUDGES.md §7:

    rule violations   OPEN -> FIXING... -> FIXED | STOOD ITS GROUND
    pathologies       OPEN -> FIXING... -> REPAIRED | SURVIVED | INDUCED

Everything is stamped OPEN today: the straight pipeline has no rounds yet.
The pathology register is empty until its detectors land (next build).

A ticket never moves a score — the numbers were computed before it existed.
It is a pure function of the saved Stage 4 record, so an old run gets its
register the moment it is re-opened.
"""
from __future__ import annotations

from typing import Any

# Code findings below this severity are bookkeeping (stringified lists, label
# instead of id); they stay in their panels and do not earn a ticket.
MIN_CODE_SEVERITY = 1

JUDGE_SOURCES = ("card_judge", "ab_judge", "runoff_judge")


def _t(register: str, source: str, kind: str, evidence: str, *,
       rank: Any = None, severity: int | None = None, advisory: bool = False,
       extra: dict | None = None) -> dict[str, Any]:
    t = {"register": register, "source": source, "kind": kind,
         "evidence": evidence, "rank": rank, "severity": severity,
         "advisory": advisory, "status": "open", "round": 0}
    if extra:
        t.update(extra)
    return t


def _code_tickets(s4: dict) -> list[dict]:
    out: list[dict] = []
    conf = s4.get("conformance") or {}
    for i in conf.get("issues") or []:
        if not isinstance(i, dict):
            continue
        if int(i.get("severity") or 0) < MIN_CODE_SEVERITY:
            continue
        where = i.get("graph") or "card"
        out.append(_t("rule", f"code · {where}", str(i.get("rule") or i.get("category")),
                      str(i.get("detail") or ""), rank=i.get("rank"),
                      severity=int(i.get("severity") or 0)))
    ia = s4.get("internal_alignment") or {}
    for f in ia.get("failures") or []:
        if not isinstance(f, dict):
            continue
        if int(f.get("severity") or 0) < MIN_CODE_SEVERITY:
            continue
        out.append(_t("rule", "code · internal", str(f.get("category") or f.get("rule")),
                      str(f.get("detail") or ""), rank=f.get("rank"),
                      severity=int(f.get("severity") or 0)))
    for e in (s4.get("trust") or {}).get("singular_errors") or []:
        if not isinstance(e, dict) or e.get("waived"):
            continue
        out.append(_t("rule", "code · error library", str(e.get("id")),
                      str(e.get("detail") or ""),
                      severity=3 if float(e.get("deduction") or 0) >= 0.2 else 2,
                      extra={"deduction": e.get("deduction")}))
    return out


def _judge_tickets(s4: dict) -> list[dict]:
    """A judge's verdict becomes a ticket only when it asks for something —
    a clean verdict is not a finding."""
    out: list[dict] = []
    cj = s4.get("card_judge") or {}
    for f in (cj.get("rollup") or {}).get("findings") or []:
        if not isinstance(f, dict):
            continue
        votes = f"{f.get('votes')}/{f.get('n')}" if f.get("n") else ""
        out.append(_t("rule", "judge · card", str(f.get("kind")),
                      f"{f.get('text')} ({votes}{', thin' if f.get('thin') else ''})",
                      rank=f.get("rank"), advisory=True))
    gj = s4.get("graph_judge") or {}
    text = gj.get("text") or {}
    acc = (text.get("account") or {})
    if acc.get("verdict") == "account_b":
        out.append(_t("rule", "judge · A-vs-B", "advice_on_the_weaker_account",
                      "asked independently, the model describes the scene "
                      "better than its advice implies — the advice rests on "
                      f"the weaker account ({acc.get('votes')}/{acc.get('n')}"
                      + ("" if gj.get("twins_agree") in (None, True)
                         else ", twins disagree") + ")",
                      advisory=True))
    vic = text.get("victims") or {}
    if vic.get("verdict") == "account_b":
        sets = gj.get("sets") or {}
        out.append(_t("rule", "judge · A-vs-B", "advice_protects_the_lesser_set",
                      "the model's own belief names the set in more danger "
                      f"({', '.join(sets.get('graph_b') or [])}); the advice "
                      f"acts as if endangered: {', '.join(sets.get('graph_a') or [])}",
                      advisory=True))
    ro = s4.get("runoff_judge") or {}
    for app, v in ro.items():
        if not isinstance(v, dict) or not v.get("text"):
            continue
        verdict = (v.get("text") or {}).get("verdict")
        if verdict not in ("answer_a", "answer_b"):
            continue
        cand = v.get("candidate_a" if verdict == "answer_a" else "candidate_b")
        first = str(cand or "").strip().splitlines()[0:1]
        out.append(_t("rule", f"judge · runoff · {app}", "model_stands_behind_another_answer",
                      f"of the answers it gave when re-asked, the model stands "
                      f"behind {'A' if verdict == 'answer_a' else 'B'}: "
                      f"{first[0].strip() if first else '?'}"
                      + ("" if v.get("twins_agree") in (None, True)
                         else " (twins disagree)"),
                      advisory=True,
                      extra={"application": app}))
    return out


def stage4_register(s4: dict | None) -> dict[str, Any]:
    """{rule: [tickets], pathology: [tickets], counts}. Pure; never raises on a
    partial record."""
    s4 = s4 or {}
    rule = _code_tickets(s4) + _judge_tickets(s4)
    for n, t in enumerate(rule):
        t["id"] = f"r{n + 1}"
    pathology: list[dict] = []
    for n, tk in enumerate((s4.get("pathology") or {}).get("tickets") or []):
        if not isinstance(tk, dict):
            continue
        pathology.append(_t("pathology", f"detector · {tk.get('technique', '')}",
                            str(tk.get("pathology")), str(tk.get("evidence", "")),
                            advisory=True,
                            extra={"strength": tk.get("strength"),
                                   "recs": tk.get("recs"),
                                   "direction": tk.get("direction"),
                                   "findings": tk.get("findings"),
                                   "subtype": tk.get("subtype"),
                                   "summary": tk.get("summary"),
                                   "pathology_kind": tk.get("kind"),
                                   "single_baseline": tk.get("single_baseline"),
                                   "id": f"p{n + 1}"}))
    return {"rule": rule, "pathology": pathology,
            "counts": {"rule": len(rule), "pathology": len(pathology),
                       "judge": sum(1 for t in rule if t["advisory"]),
                       "code": sum(1 for t in rule if not t["advisory"])}}
