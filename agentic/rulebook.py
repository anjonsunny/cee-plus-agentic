"""The perception rulebook: rules as retrievable chunks (RAG seam v0).

WHY THIS FILE EXISTS
====================
Loop 1's repair messages used to carry hand-authored instruction strings
buried in detect_violations. Agreed with Sunny (2026-07-21): when a
violation fires we know its kind, and that IS a retrieval query; the
repair message should carry the retrieved rule, its rationale, and a
worked example, so the model is corrected by the rulebook itself, not by
ad-hoc phrasing.

Each rule is ONE chunk with its statement, rationale, worked example, and
instruction template intact (chunking on rule boundaries; a rule split
across chunks is a broken rule). This mirrors how the future full-corpus
rulebook will be chunked for LlamaIndex + Chroma.

RETRIEVAL V0 IS EXACT LOOKUP, ON PURPOSE
========================================
At five rule families, keyed lookup beats embeddings (AGENTIC_PLAN Stage
22's own honesty caveat). retrieve() is the stable interface; when the
corpus grows (effect-label truth conditions, threats-stage rules join in
Stage 2/3), the body swaps to hybrid vector retrieval and callers do not
change. That swap gets a measured before/after, not silent adoption.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleChunk:
    """One rule, whole: statement + why + worked example + how to ask for
    the fix. `template` placeholders are filled with the violation's
    evidence (entity index, the offending text, legal options)."""

    rule_id: str
    rule: str            # the statement, one sentence
    rationale: str       # why the rule exists (shown to the model)
    example: str         # worked wrong -> right example
    template: str        # evidence-citing fix request


RULES: dict[str, RuleChunk] = {
    "family_name_as_label": RuleChunk(
        rule_id="P1",
        rule="A label must be one specific noun from the vocabulary; family "
             "names (vehicle, structure, hazard_media...) are categories, "
             "never labels.",
        rationale="Downstream hazard analysis depends on the specific type: "
                  "a tanker_truck and a car carry different dangers. A "
                  "category label silently erases that difference.",
        example="WRONG: {\"label\": \"vehicle\"} for a fuel tanker. "
                "RIGHT: {\"label\": \"tanker_truck\"}.",
        template="Entity {index}: '{raw_label}' is a family name, not a "
                 "label. Replace it with the specific noun that fits: "
                 "{members}. If none fits, use 'other'.",
    ),
    "label_out_of_vocab": RuleChunk(
        rule_id="P2",
        rule="Labels come from the closed vocabulary; anything else must be "
             "a deliberate 'other' with a clear description.",
        rationale="The closed vocabulary keeps entities comparable across "
                  "scenes and models; 'other' is the honest escape hatch "
                  "and is counted, so genuine gaps become visible.",
        example="WRONG: {\"label\": \"zamboni\"}. RIGHT: {\"label\": "
                "\"other\", \"description\": \"ice-resurfacing machine near "
                "the rink\"}.",
        template="Entity {index}: '{raw_label}' is not in the allowed "
                 "labels. Pick the closest allowed label, or keep it as "
                 "'other' with a clear description if nothing fits.",
    ),
    "state_out_of_vocab": RuleChunk(
        rule_id="P3",
        rule="States come from the state vocabulary; one lowercase word "
             "describing the entity's current condition.",
        rationale="States drive the causal analysis (a hazard IS a state on "
                  "an entity); an invented state word cannot be reasoned "
                  "about or suppressed downstream.",
        example="WRONG: state \"operating\" for a parked tanker. RIGHT: "
                "state \"stationary\" (or \"leaking\" if it is actively "
                "discharging).",
        template="Entity {index} ('{raw_label}'): state '{state}' is not a "
                 "vocabulary state. Choose the single word that matches "
                 "what you see: {state_words}.",
    ),
    "missing_anchor_bbox": RuleChunk(
        rule_id="P4",
        rule="Every entity carries a rough pixel bbox; it anchors WHICH "
             "instance you mean.",
        rationale="The detector refines geometry, but only your box says "
                  "which of two similar objects you meant (the burning "
                  "house, not the intact one).",
        example="WRONG: no bbox. RIGHT: \"bbox\": [420, 160, 840, 610] "
                "(approximate is fine).",
        template="Entity {index} ('{raw_label}'): missing bbox. Add a rough "
                 "pixel box [x1, y1, x2, y2] around it. Approximate is fine.",
    ),
    "caption_entity_missing": RuleChunk(
        rule_id="P5",
        rule="Entities named in the caption must appear in the list, or be "
             "explicitly un-seeable in the image.",
        rationale="The caption is given input, not opinion. A named entity "
                  "with no list entry is checkable evidence of a dropped "
                  "entity; standing your ground is allowed and recorded.",
        example="Caption says 'a tanker truck leaks fuel' -> the list needs "
                "a tanker_truck entity and a spill entity, or your "
                "unchanged list states you cannot see them.",
        template="The caption mentions \"{raw_phrase}\" but your list has "
                 "no '{wanted}' entity. Look at the image again and add it "
                 "with its state and bbox. If you truly cannot see it in "
                 "the image, leave your list unchanged.",
    ),
}


def retrieve(kind: str) -> RuleChunk | None:
    """The retrieval interface. V0: exact lookup by violation kind (the
    right tool at 5 rules). Future: hybrid vector retrieval over the full
    rulebook corpus; same signature, measured swap."""
    return RULES.get(kind)


def instruction_for(kind: str, **evidence) -> str:
    """Compose a repair instruction: the evidence-citing fix request, then
    the retrieved rule, rationale, and worked example. This is the text a
    ticket shows and the model receives; the rulebook speaks, not ad-hoc
    phrasing."""
    chunk = retrieve(kind)
    if chunk is None:
        return f"Violation '{kind}': {evidence}"
    ask = chunk.template.format(**evidence)
    return (f"{ask}\n  Rule {chunk.rule_id}: {chunk.rule}\n"
            f"  Why: {chunk.rationale}\n  Example: {chunk.example}")
