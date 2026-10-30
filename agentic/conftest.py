"""Hermetic test settings for agentic/."""
import os

# Arm A's embedding measure (rec_semantic_shift) loads a sentence-transformer;
# the hermetic suite must never load a model. The live pipeline computes it.
os.environ.setdefault("CEE_DISABLE_SEMANTIC", "1")

# Box convention is a property of the live subject (Qwen3 family: 0-1000
# grid). Hermetic fixtures hand in pixel boxes, so pin pixels here; the
# grid tests set the variable themselves.
os.environ.setdefault("SUBJECT_BBOX_GRID", "0")
