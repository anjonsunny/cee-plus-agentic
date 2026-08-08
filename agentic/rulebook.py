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
                 "label. Your own description of this entity was: "
                 "\"{description}\". Replace the label with the ONE most "
                 "specific noun from the allowed vocabulary that matches "
                 "that description — from ANY family, not only "
                 "'{raw_label}' (whose members are: {members}). If "
                 "nothing in the vocabulary truly fits, use 'other'.",
    ),
    "duplicate_entity": RuleChunk(
        rule_id="P6",
        rule="One real-world individual gets ONE entry. Two entries of "
             "the same living kind whose boxes lie almost exactly on top "
             "of each other are, absent an explicit reason, the same "
             "individual listed twice.",
        rationale="Phantom duplicates corrupt everything counted "
                  "downstream: geometry pairs, membership votes, at-risk "
                  "lists, and ultimately how many lives the scene "
                  "contains. Merging is not erasure — the individual "
                  "stays in the record under its better label.",
        example="WRONG: a generic person entry AND a specific-role entry with near-identical "
                "boxes for one officer. RIGHT: a single police_officer "
                "entry (the more specific label wins).",
        template="Entities {index_a} ('{label_a}': \"{desc_a}\") and "
                 "{index_b} ('{label_b}': \"{desc_b}\") have boxes that "
                 "overlap almost entirely (IoU {iou}). If they are the "
                 "same individual, keep ONE entry with the more specific "
                 "label. If they are truly two different individuals, "
                 "keep both and make each description clearly point at a "
                 "different person.",
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
                 "vocabulary state. The legal words, grouped by kind, are "
                 "{state_words}.\n"
                 "Pick the ONE word that matches THIS entity's own "
                 "condition, not the scene's overall situation.",
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
             "explicitly un-seeable in the image. A caption's mention of an "
             "ATTACHED medium is already satisfied by the entity carrying "
             "that state: 'a house on fire' is satisfied by a house with "
             "state 'burning' — no separate fire entity is owed (fluid "
             "convention). Free-burning fire ('a brush fire'), smoke, dust, "
             "gas, and spills are their own entities and stay required.",
        rationale="The caption is given input, not opinion. A named entity "
                  "with no list entry is checkable evidence of a dropped "
                  "entity; standing your ground is allowed and recorded. "
                  "But the completeness check must not contradict the fluid "
                  "convention by demanding an entity for a medium the list "
                  "already expresses as an attached state.",
        example="Caption says 'a tanker truck leaks fuel' -> the list needs "
                "a tanker_truck entity and a spill entity, or your "
                "unchanged list states you cannot see them. Caption says "
                "'a house on fire' and the list has a burning house -> "
                "nothing is missing.",
        template="The caption mentions \"{raw_phrase}\" but your list has "
                 "no '{wanted}' entity. Look at the image again and add it "
                 "with its state and bbox. If you truly cannot see it in "
                 "the image, leave your list unchanged.",
    ),
    "state_is_a_label": RuleChunk(
        rule_id="P3b",
        rule="The state slot takes a CONDITION, never a noun. A word that is "
             "already a legal label names what the entity IS; the state names "
             "what condition it is in right now.",
        rationale="The two slots ask different questions and one vocabulary "
                  "answers both, so a noun in the state slot is legal-looking "
                  "and meaningless. The generic vocabulary reminder does not "
                  "help here: the model did not reach for an unknown word, it "
                  "answered 'what is it?' twice. Naming that confusion is "
                  "cheaper and more precise than listing thirty legal words "
                  "again — and it says nothing about which condition is true, "
                  "which stays the model's judgement.",
        example="A noun repeated in the state slot -> flagged as a slot "
                "confusion, not as an unknown word. A genuine condition the "
                "vocabulary lacks -> the ordinary out-of-vocab rule.",
        template="Entity {index} ('{raw_label}'): '{state}' is WHAT THIS "
                 "ENTITY IS, not what condition it is in — you have already "
                 "said that in the label. The state slot asks for its "
                 "physical condition right now. The legal words, grouped by "
                 "kind, are {state_words}.\n"
                 "Pick the ONE that matches what you can see of THIS "
                 "entity's condition.",
    ),
    "caption_state_contradiction": RuleChunk(
        rule_id="P7",
        rule="A condition word the caption uses must be accounted for in the "
             "declared states. P5 checks whether a caption's ENTITY is "
             "missing; this checks whether its CONDITION is. When the "
             "caption describes a dangerous condition and no entity carries "
             "that condition as its state, the list and the given text "
             "disagree.",
        rationale="A state word can be legal and still be wrong. When it is, "
                  "the vocabulary check passes it — the word is in the "
                  "vocabulary — and the completeness check passes it too, "
                  "because the entity itself is present. A mistaken condition "
                  "then enters the record unopposed and every later stage "
                  "inherits it. Every other pressure signal here is a "
                  "SELF-consistency signal: probes, triggers and petitions all "
                  "ask the answer about itself, and an answer that is wrong "
                  "the same way each time satisfies all of them. This check is "
                  "the only one that reads a source outside the answer — the "
                  "caption supplied with the task. It quotes that caption's "
                  "own words and never names which state is right; leaving "
                  "the list unchanged is a legal outcome and is recorded.",
        example="A condition named in the caption that no entity carries as a "
                "state -> the disagreement is raised for reconciliation. A "
                "condition the declared states already express -> nothing "
                "fires. A caption word that is not a state at all is raised "
                "the same way and is dismissed by leaving the list unchanged.",
        template="The caption uses condition words that none of your entities "
                 "carries as a state: {word}. Your declared states are: "
                 "{declared}. Look at the image again and reconcile your list "
                 "with the caption. Some of these may not be states at all — "
                 "if your list is right, leave it unchanged.",
    ),
    # ── Stage 2 scene-assessment rules (S-family, added for the Stage 2
    # loop). Worked examples are drawn from the six frozen scenes — the
    # calibration set Sunny designated as working GT. ──────────────────
    "threat_reason_victim_shaped": RuleChunk(
        rule_id="S8",
        rule="A threat's reason must describe harm the entity CAUSES, "
             "never harm it receives. An entity justified by its own "
             "endangerment does not belong in the threat slot.",
        rationale="The slot and the story contradict, and the story is "
                  "your honest belief: if the entity is endangered, it is "
                  "at-risk, not a threat. This shape usually appears when "
                  "a check is being paid off rather than answered — the "
                  "promotion (and possibly the entity's declared state) "
                  "is in doubt. Two legal exits: remove the entity from "
                  "threats, or stand by its hazardous state and say what "
                  "it actively harms. If you doubt the declared state "
                  "itself, keep the entity out of threats and say so in "
                  "'reasoning' — that dispute is recorded, not punished.",
        example="WRONG: a road listed as a threat because it 'could be at "
                "risk if the fire spreads' — that sentence describes a "
                "victim. RIGHT: either the road leaves the threat list, "
                "or the reason names what the road actively harms and by "
                "what mechanism.",
        template="{evidence}. Either remove it from threats, or justify "
                 "what it actively harms and how; if you doubt its "
                 "declared state, keep it out of threats and record the "
                 "dispute in 'reasoning'.",
    ),
    "scenario_level_incoherent": RuleChunk(
        rule_id="S1",
        rule="The verdict must agree with its own severity: scenario 'No' "
             "requires level 0 and type N/A; scenario 'Yes' requires "
             "level >= 1.",
        rationale="A verdict that contradicts its own number cannot be "
                  "used by any downstream stage: Stage 12 reads the level "
                  "as a shift basis and the scenario as its gate.",
        example="WRONG: {\"disaster_scenario\": \"No\", \"disaster_level\": "
                "3}. RIGHT: either No with level 0, or Yes with level 3.",
        template="Your verdict is internally contradictory: {evidence}. "
                 "Re-issue a coherent verdict (No requires level 0 and "
                 "type N/A; Yes requires level >= 1).",
    ),
    "missed_disaster_incoherence": RuleChunk(
        rule_id="S2",
        rule="If the declared perception contains hazard-bearing or "
             "at-risk states, a 'No / level 0' verdict contradicts "
             "upstream state and must be justified or revised.",
        rationale="The verdict must follow from declared perception. "
                  "Dismissing declared hazards without addressing them is "
                  "the failure shape where a scene carrying danger states is "
                  "assessed as no-disaster.",
        example="WRONG: declared entities include a leaking tanker and a "
                "spreading fire, yet the verdict says No/0. RIGHT: Yes, "
                "with a level that reflects the active hazards.",
        template="Your 'No' verdict contradicts the declared perception: "
                 "{hazard_states}. Either revise the verdict to account "
                 "for these states, or explain concretely why they do not "
                 "constitute a disaster.",
    ),
    "false_alarm_incoherence": RuleChunk(
        rule_id="S3",
        rule="A 'Yes' verdict requires support in the declared state: at "
             "least one hazard-bearing or at-risk entity state, or an "
             "explicit hazard in the caption.",
        rationale="The control scene exists to test exactly this: a "
                  "disaster verdict conjured from a calm scene is a false "
                  "alarm, and unsupported escalation is as measurable a "
                  "failure as a miss.",
        example="WRONG: a Yes/level-4 verdict on a calm scene where every "
                "entity is stationary or standing and the caption reports "
                "nothing hazardous. RIGHT: No · N/A · 0.",
        template="Your 'Yes' verdict has no support in the declared "
                 "perception: no hazard-bearing or at-risk states, and "
                 "the caption reports nothing hazardous. Either point to "
                 "the specific declared evidence you relied on, or revise "
                 "to 'No'.",
    ),
    "hazard_not_in_threats": RuleChunk(
        rule_id="S6",
        rule="On a 'Yes' verdict, every declared hazard-bearing entity "
             "belongs in threats — or the answer must argue concretely "
             "why it is not currently a source of harm.",
        rationale="A declared hazard the threat list forgets can never be "
                  "suppressed at Stage 8 and silently corrupts coverage. "
                  "Forgetting is invisible; arguing is auditable.",
        example="A seeping spill is declared, but the threats list names "
                "only the tanker and the fire — the spill is the "
                "spreading medium and belongs in threats too.",
        template="{object_id} carries hazard-bearing state '{state}' but "
                 "your threats list omits it. Add it with a reason, or "
                 "state concretely why it is not a current harm source.",
    ),
    "hazard_as_at_risk": RuleChunk(
        rule_id="S7",
        rule="A non-living entity in BOTH threats and at_risk breaches "
             "the one-state-one-role schema. LIVING BEINGS are the "
             "standing exception: a burning or contagious person is a "
             "source of harm to others AND a victim simultaneously — "
             "list them in both, kind=distress, no argument needed.",
        rationale="One declared state maps to one causal role. Most "
                  "double-listings are confusion (a burning house 'at "
                  "risk of the fire' makes suppression circular). Genuine "
                  "dual roles exist (a contagious person harms others "
                  "WHILE being sick) but the honest resolution is to "
                  "argue it concretely — or split the entity, as the "
                  "fluid convention splits tanker and spill.",
        example="WRONG: a burning house also listed in at_risk — its "
                "intact neighbor is the victim, list that instead. RIGHT: "
                "a burning PERSON in threats AND at_risk (distress) — a "
                "living being in a hazard state is both source and victim.",
        template="{object_id} carries hazard-bearing state '{state}' and "
                 "also appears in at_risk. Either remove it from at_risk "
                 "(listing the NEIGHBORING endangered entity instead), or "
                 "keep both and argue the dual role concretely (how it "
                 "harms others AND is itself harmed). A standing, argued "
                 "dual role is recorded, not overruled.",
    ),
    # ── Merged-stage rules: ids and states (S4, S5) ────────────────────
    "id_not_in_perception": RuleChunk(
        rule_id="S4",
        rule="Every object_id cited in threats or at_risk must exist in "
             "detected_objects; the declared perception is the only "
             "universe of entities.",
        rationale="A cited id that resolves to nothing is an invented "
                  "entity — the exact grounding leak this pipeline exists "
                  "to make impossible.",
        example="WRONG: citing an id that appears nowhere in "
                "detected_objects. RIGHT: cite only the declared ids, or "
                "nothing.",
        template="Your answer cites '{object_id}', which is not in the "
                 "declared perception ({known}). Cite only listed ids, or "
                 "drop the claim.",
    ),
    "threat_state_not_hazardous": RuleChunk(
        rule_id="S5",
        rule="A threat must carry a hazard-bearing state (burning, "
             "leaking, spreading...); an entity in a normal state is not "
             "a source of harm.",
        rationale="A hazard IS a state on an entity. Calling a standing "
                  "person a threat unmoors the causal analysis that "
                  "suppression will later act on.",
        example="WRONG: a standing bystander or a drowning child listed "
                "as a threat. RIGHT: the burning structure is the threat "
                "— or threats=[] : a victim in distress with NO visible "
                "active hazard is a legitimate scene shape; never invent "
                "a source.",
        template="You listed {object_id} as a threat, but its declared "
                 "state '{state}' is {state_kind}, not hazard-bearing. "
                 "Cite an entity whose state is the harm source, or drop "
                 "it.",
    ),
    # ── Geometry rules (G-family, added with the Stage 2+3 merge) ──────
    "geometry_adjacency": RuleChunk(
        rule_id="G1",
        rule="An entity whose box overlaps or sits adjacent to an active "
             "hazard's box is a PROXIMITY at-risk CANDIDATE.",
        rationale="Danger spreads through space; nearness to an active "
                  "hazard is the evidence that nominates an entity even "
                  "when its own state is normal.",
        example="A standing person adjacent to a burning structure "
                "(small pixel gap) -> candidate. The model then decides "
                "and cites both the state and the adjacency.",
        template="Geometry nominates {object_id} ({state}) as "
                 "{relation} to {hazard}. Decide whether it is at risk; "
                 "cite the state and the spatial evidence either way.",
    ),
    "proximity_without_hazard": RuleChunk(
        rule_id="G2",
        rule="Proximity at-risk requires an ACTIVE hazard to be near: no "
             "hazard-bearing entity in the scene means no proximity "
             "at-risk.",
        rationale="Risk needs a source. Proximity to nothing is the "
                  "false-alarm shape the control scene exists to catch.",
        example="WRONG: a parked car marked 'proximity' at-risk on a "
                "calm scene. RIGHT: at_risk=[] when no hazard exists.",
        template="You marked {object_id} as proximity at-risk, but the "
                 "declared perception contains no hazard-bearing entity. "
                 "Name the hazard it is near, or drop the claim.",
    ),
    "geometry_is_a_hint": RuleChunk(
        rule_id="G3",
        rule="Box adjacency is 2D image-plane evidence: it NOMINATES "
             "proximity candidates but never convicts; real closeness is "
             "3D and may need a visual check.",
        rationale="A person 50 meters behind a house overlaps its box. "
                  "Treating 2D adjacency as proof would manufacture "
                  "at-risk entities out of perspective accidents.",
        example="An animal's box touches a burning structure's box, but "
                "the animal is across the street behind it -> geometry "
                "nominates, the model (or a look_at check) decides.",
        template="The adjacency of {object_id} to {hazard} is a 2D hint "
                 "only. Weigh it against the states and caption; if still "
                 "ambiguous, say so rather than guessing.",
    ),
    "at_risk_kind_mismatch": RuleChunk(
        rule_id="G4",
        rule="The entity's declared state ALONE decides the at-risk kind "
             "(baseline main.py:26, schema-strict): at-risk vocabulary -> "
             "distress; normal vocabulary -> proximity. Both directions "
             "are errors: downgrading a drowning child to proximity, or "
             "promoting a standing bystander to distress.",
        rationale="Distress means harm is ACTUALIZED (present tense); "
                  "proximity means harm is IMMINENT (potential). The "
                  "causal graph reads tense from the state, and "
                  "suppression plans different actions for each — a "
                  "mislabeled kind corrupts the chain downstream.",
        example="WRONG: a standing person marked distress for 'potential "
                "exposure' — potential danger from nearness IS proximity. "
                "WRONG: a drowning child marked proximity. RIGHT: the "
                "state decides.",
        template="{object_id}: your claimed kind '{given_kind}' "
                 "contradicts its declared state '{state}'. State alone "
                 "decides the kind (G4): at-risk state -> distress, "
                 "normal state -> proximity.",
    ),

    # ── R-family (F24): one recommendation, three surfaces, one law ─────
    #
    # The action is the thing being explained. The prose reason and the
    # structured quad are two INDEPENDENT explanations of it, held to the same
    # constraints and then made to answer for each other. These chunks are the
    # text half of that law — code detects, the rulebook teaches — and become
    # the tickets Stage 4 reflection quotes back.
    "action_names_no_object_id": RuleChunk(
        rule_id="R1",
        rule="An action names the entities it operates on by their object_id, "
             "never by a prose description of them.",
        rationale="A description cannot be checked against anything. Two "
                  "readers can disagree about which entity 'the area' meant, "
                  "and so can the quad — which is exactly the disagreement "
                  "the quad exists to rule out. An id makes the action "
                  "answerable.",
        example="WRONG: 'Secure the fallen vehicle.' RIGHT: 'Secure "
                "<that vehicle's object_id>.' Same action, now checkable.",
        template="Recommendation {rank}: your action reads '{action}'. It "
                 "names no object_id, so nothing ties it to the danger you "
                 "declared. Rewrite it naming the object_ids it operates on, "
                 "exactly as they appear in the scene list. Do not change "
                 "which action you are recommending.",
    ),
    "quad_explains_a_different_action": RuleChunk(
        rule_id="R2",
        rule="The action comes first; the reason and the quad are written to "
             "explain it. So each explanation must account for what the "
             "action actually operates on — the threat it addresses, or "
             "someone that threat harms.",
        rationale="This is the declared-vs-operative split, and the direction "
                  "of it matters: an explanation that covers none of what the "
                  "action touches has not been written for that action. A "
                  "responder can name the serious hazard and then do "
                  "something about a lesser one, and the write-up still reads "
                  "correct. Making each explanation answer for the action is "
                  "what turns that from persuasive into visible.",
        example="WRONG: the action stabilises one object while the quad "
                "declares an entirely different hazard that never touches it. "
                "RIGHT: the explanation names the danger that the action is "
                "actually responding to.",
        template="Recommendation {rank}: your action operates on {acted_on}. "
                 "Your quad declares {threat} and names {covered} — none of "
                 "which is what the action touches, so the quad is explaining "
                 "a different recommendation. Rewrite the quad to explain "
                 "THIS action, or say that the action was the wrong one.",
    ),
    "reason_threat_not_declared": RuleChunk(
        rule_id="R3",
        rule="The prose reason obeys the SAME rules as the quad: its subject "
             "must be a declared threat, its state a declared state, its verb "
             "an effect from the list, and at-risk roles are never states.",
        rationale="Holding the structure to a law and letting the prose run "
                  "free guarantees they diverge, and then we cannot tell a "
                  "real disagreement from one the rules forced. Under ONE law "
                  "a surviving mismatch is the model's, and it means "
                  "something.",
        example="WRONG: prose blames an entity that is not on the threats "
                "line, so the quad has to substitute a different one. RIGHT: "
                "the prose names a declared threat, and the quad is that same "
                "sentence with its slots filled.",
        template="Recommendation {rank}: your reason says '{reason}'. It "
                 "blames {reason_threat}, which is not on the threats line "
                 "for this scene. Either name a declared threat, or say that "
                 "your quad's threat was the one you meant.",
    ),
    "subject_mismatch": RuleChunk(
        rule_id="R4",
        rule="The reason and the quad are the same claim in two forms. The "
             "reason's subject IS the quad's threat; the entities the reason "
             "says are harmed ARE the quad's affected_objects.",
        rationale="Sharing the same object_ids is not agreeing. The same two "
                  "entities can appear in both with cause and effect swapped, "
                  "and an id-overlap check passes it. Comparing the ROLES is "
                  "what catches a reversed direction — the error CEE+ exists "
                  "to find.",
        example="WRONG: the prose has A harming B while the quad has B "
                "harming A — identical ids, opposite claim. RIGHT: both name "
                "the same source, the same effect, the same harmed entities.",
        template="Recommendation {rank}: your reason blames "
                 "{reason_threat} while your quad blames {quad_threat}. Those "
                 "are two different causal claims about one action. Decide "
                 "which one you mean and make both say it.",
    ),
    "remaining_risk_role_word": RuleChunk(
        rule_id="R5",
        rule="remaining_risk is an (object_id, state) pair. 'distress' and "
             "'proximity' are at_risk_as ROLES, not states — the same ban the "
             "quad already carries.",
        rationale="One vocabulary law across every slot that takes a state. A "
                  "role word here says how an entity is exposed, not what "
                  "condition it is in, so the pair names no residual hazard "
                  "and the field silently stops meaning anything.",
        example="WRONG: a pair whose second half is an exposure role. RIGHT: "
                "a pair whose second half is that entity's declared state.",
        template="Recommendation {rank}: your remaining_risk is "
                 "'{remaining_risk}'. '{role}' is how an entity is exposed, "
                 "not a condition it is in. Give the entity's declared state "
                 "instead, or name a different residual risk.",
    ),
}

# Violation kinds that share a rule: the two internal-check directions
# are both faces of S1.
_ALIASES = {
    "scenario_no_level_gt0": "scenario_level_incoherent",
    "scenario_yes_level_0": "scenario_level_incoherent",
    "distress_downgraded": "at_risk_kind_mismatch",   # old name, same rule
    "scenario_no_with_entities": "scenario_level_incoherent",  # S1 face
}


def retrieve(kind: str) -> RuleChunk | None:
    """The retrieval interface. V0: exact lookup by violation kind (the
    right tool at 8 rules). agentic/rulebook_rag.py holds the LlamaIndex +
    Chroma semantic index over the SAME chunks for query-shaped lookups;
    kind-shaped lookups stay exact — a violation kind is a key, not a
    similarity search."""
    return RULES.get(_ALIASES.get(kind, kind))


def _compose_instruction(chunk: RuleChunk | None, kind: str,
                         evidence: dict) -> str:
    """Format the fix request + rule text. Robust to a template whose
    placeholders don't match the evidence — that happens ONLY in 'rag'
    mode, when RAG returns a different rule than the kind that fired; we
    then skip the fill-in and just teach the (wrongly-picked) rule."""
    if chunk is None:
        return f"Violation '{kind}': {evidence}"
    try:
        ask = chunk.template.format(**evidence)
    except (KeyError, IndexError):
        ask = f"Problem ({kind}): {evidence}"
    return (f"{ask}\n  Rule {chunk.rule_id}: {chunk.rule}\n"
            f"  Why: {chunk.rationale}\n  Example: {chunk.example}")


def instruction_for(kind: str, **evidence) -> str:
    """Compose a repair instruction: the evidence-citing fix request, then
    the retrieved rule, rationale, and worked example. The rule is fetched
    through the retrieval switch (agentic/retrieval.py), so repair's
    lookups honor rulebook|rag|both and join the RAG shadow. Default
    'rulebook' mode returns the exact-key rule — byte-identical to before,
    so repair behavior (and the LangGraph equivalence) is unchanged."""
    from agentic.retrieval import retrieve_rule  # lazy: avoids import cycle
    chunk, _meta = retrieve_rule(kind)
    return _compose_instruction(chunk, kind, evidence)
