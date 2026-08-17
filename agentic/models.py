"""One place that says which model sits in which seat.

The system is MODEL-AGNOSTIC (Sunny, 2026-08-08): any VL model can be the
subject, any capable model can judge — swapping a seat is an edit here (or an
env var), never a hunt through call sites. What agnosticism costs is honesty
in the record: every run and every capture record is stamped with the ids
that produced it, because a directory of runs from mixed subjects is
unusable without the stamp.

THE SEATS

  SUBJECT       the model under measurement. The specimen, not a component
                to optimize — "which subject" is a research choice, and the
                instrument must hold ANY of them.
  JUDGE         the text-only judge: the OFFICIAL verdict on every
                subjective question. Deliberately a different family from
                the subject (self-preference risk).
  JUDGE_VISION  the image-aware twin. Same PROMPT as the judge, plus the
                image as context. Ideally the SAME MODEL as JUDGE so the
                image is the only variable between the twins — with
                gemma4 both seats hold one model, which is the clean design.
  DIALOGUE      explainer / "ask the analyst" chat. Text-only on purpose.

FREEZE THE JUDGES, VARY THE SUBJECT. The judges are the measuring stick:
once a judge model passes its discrimination sets it stays fixed while
subjects rotate, or cross-subject comparisons confound the specimen with
the instrument. Changing a judge seat means re-running every discrimination
set before its verdicts count.

HISTORY
  through 2026-08-08   subject qwen2.5vl:7b · judge llama3.1:8b (text-only,
                       no vision twin existed). Every run in
                       exports/agentic_runs/ before this date is that pair.
  from 2026-08-08      subject qwen3-vl:8b · judge twins gemma4:26b.
                       Decided with the registry search of 2026-08-08:
                       no Qwen3.5-VL exists yet, Mistral's 2026 vision
                       models (~120B) do not fit the 48 GB machine, and
                       gemma4:26b (Apr 2026, MoE, 3.8B active) is the
                       newest vision-capable judge that fits beside the
                       subject.
"""
from __future__ import annotations

import os

SUBJECT_MODEL = os.getenv("SUBJECT_MODEL",
                          os.getenv("QWEN_MODEL_NAME", "qwen3-vl:8b"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:26b")
JUDGE_VISION_MODEL = os.getenv("JUDGE_VISION_MODEL", JUDGE_MODEL)
DIALOGUE_MODEL = os.getenv("DIALOGUE_MODEL", "qwen2.5:7b")


def stamp() -> dict:
    """The id block every run record and capture record carries."""
    return {"subject_model": SUBJECT_MODEL, "judge_model": JUDGE_MODEL,
            "judge_vision_model": JUDGE_VISION_MODEL}
