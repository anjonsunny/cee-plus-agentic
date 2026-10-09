"""Hermetic test settings for agentic/."""
import os

# Arm A's embedding measure (rec_semantic_shift) loads a sentence-transformer;
# the hermetic suite must never load a model. The live pipeline computes it.
os.environ.setdefault("CEE_DISABLE_SEMANTIC", "1")
