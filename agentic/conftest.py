"""Hermetic test settings for agentic/."""
import os

# Arm A's embedding measure (rec_semantic_shift) loads a sentence-transformer;
# the hermetic suite must never load a model. The live pipeline computes it.
os.environ.setdefault("CEE_DISABLE_SEMANTIC", "1")

# Box convention is a property of the live subject (Qwen3 family: 0-1000
# grid). Hermetic fixtures hand in pixel boxes, so pin pixels here; the
# grid tests set the variable themselves.
os.environ.setdefault("SUBJECT_BBOX_GRID", "0")


import pytest


@pytest.fixture()
def three_part_shift(monkeypatch):
    """For tests that check what is DONE with a shift of a given size (the
    scores, the shares, the words): the shift is the older three parts, as
    the test's helper states them. The graph part without the state and the
    state part (2026-11-23) have their own tests."""
    from agentic import declared4
    monkeypatch.setattr(declared4, "graph_part", lambda before, after: None)
    monkeypatch.setattr(declared4, "state_part", lambda before, after: None)
