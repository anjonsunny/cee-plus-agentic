# Arm B — Findings Ledger

## FIX TAXONOMY (living table — categorize every new finding on entry)

| Cat | Failure mode | Findings | Count |
|---|---|---|---|
| A | LANGUAGE GAP — schema can't hear the model's honest English | F8, F11, F12, F10(paved), F45 | 5 |
| B | INDUCED ERROR — model was right until our machinery pressured it | F1, F2, F5, F10, F25, F27 | 6 |
| C | OUR RULES COLLIDE — our own rules fighting each other | F3, F9, F24 | 3 |
| D | JUDGE NOISE / BIAS — ill-posed questions, severity-minimizing | F4(open), F5, F11, F26, F28 | 5 |
| E | GENUINE MODEL ERROR — unstable second looks; flat self-confidence; reflection jitter | F7(parts), jitter | ~2 |
| F | METRIC DEFECT — scoring that hides/distorts the real signal | F15, F24, F25, F29, F45 | 5 |

**Standing observation (through F12, 4 of 6 scenes):** only ~2 of ~15
defects were the subject model failing unprompted. The dominant modes
are OURS (A + B = 8). The model's true deficit list: (1) capitulates
under any authoritative pressure — the thread through every B incident;
(2) second looks lose entities already found; (3) self-reported
confidence is flat 0.95, informationless. First-look perception has
been good on every scene so far. For the paper: fixing the model mostly
means fixing the INTERVIEW, not the witness.

---

Running record of research findings produced by the agentic conversion.
Each entry: what happened, the evidence trail, what changed because of
it, and status. This file is the paper's raw material.

---

## F1 · Reflection-induced capitulation
**Date:** 2026-07-22 · **Scene:** B_pool · **Status:** fixed, verified

Told "your scenario is unstable" (a scenario_flip trigger), the subject
resolved the doubt by folding: `Yes · drowning · 9` became
`No · N/A · 0` — on a drowning scene — while its own answer still listed
two children in distress. The run ended "clean" because S2/S3 existed
only as rulebook text, not code. ΔU did not flag it (0.25 → 0.225).

**Fix:** S2/S3/emptiness checks in code; capitulation guard (doubt
phrased as evidence, never as the model's reliability). **Verified:**
next run held `Yes · 9` through 2 rounds; regression test replays the
exact failure.

**Claim for the paper:** evidence-triggered reflection is not
automatically safe; doubt-framing alone can flip a correct verdict.

---

## F2 · Prompt-repair side effects
**Date:** 2026-07-22 · **Scene:** A_fire (live runs) · **Status:** fixed, verified

The distress-emphasis block added to the perception prompt (to fix
B_pool's missed second child) caused a deterministic hallucination on
A_fire: `person_1 · drowning` beside a burning house, identically on
every temp-0 run. The emphasized worked example ("drowning") leaked into
unrelated scenes.

**Fix:** medium-bound distress wording ("drowning requires water; a
person merely standing near a hazard is standing"). **Verified:** next
run: `person_1 · standing`, proximity at-risk derived correctly.

**Claim:** prompt repairs are interventions with side effects; a fixed
calibration set that is re-run after every prompt change is the
regression harness that catches them.

---

## F3 · Baseline schema pressure explains victim-as-threat
**Date:** 2026-07-22 · **Scene:** B_pool · **Status:** understood; petition design follows it

The subject repeatedly drafted drowning children into the threat slot
(2 capped rounds, S5 each time). Root cause found in the baseline
ontology (main.py:26/104): every at-risk entry must be `affected_object`
of a recommendation quad; quads need a threat slot; self-loops are
forbidden. The designed escape is the ENGULFING medium (the pool as its
own hazard entity) — which Stage 1 never declared. The model needs a
hazard to exist; given none, it invents one from available parts.

**Consequence:** the re-perception petition for drowning-shaped scenes
should request the engulfing medium specifically, citing the fluid
convention.

---

## F4 · Judge conservatism (llama3.1:8b) — OPEN, collecting data
**Status:** 2 data points, watching

| # | Scene / run | Judge role | Verdict | vs GT |
|---|---|---|---|---|
| 1 | A_fire live (pre-evidence-basis fix) | pairwise | preferred PRE (minimal threats) | GT said POST was REFINEMENT → judge WRONG |
| 2 | A_fire live (post-fix, full evidence basis) | runoff | preferred the MINORITY `L4` reading over the 4/5 `L8-9` majority | GT scored the followed advice FALSE CERTAINTY → judge WRONG (severity-minimizing) |
| 2b | same run | pairwise | preferred PRE (`L7`, no car_1) | GT agreed → judge RIGHT |

Emerging pattern: llama3.1 as judge favors LOWER severity and SMALLER
entity sets. Note #2 vs #2b: the same judge family steered the
degradation and then correctly flagged it — advisory bias is
direction-dependent (it pushes down, and approves of down-pushes it
didn't cause being undone).

**Decision pending (Sunny):** after more scenes, either swap judge
model (e.g. mistral) or keep llama behind a substantiation gate
(advice forwarded only when the judge's reason cites a declared
entity/state). Until decided, runoff/pairwise remain ADVISORY ONLY and
every verdict lands in this table.

---

## F5 · Judge-induced false certainty
**Date:** 2026-07-22 · **Scene:** A_fire live · **Status:** open; mitigations proposed

The runoff advice (F4 #2) was taken by the subject AGAINST its own 4/5
probe majority: severity L7→L4, fire_1 dropped from threats (it was in
4/5 probe threat lists), car_1 added as a "threat" with a victim-shaped
reason. GT error 0.0 → 2.0 while U fell 0.311 → 0.289: the FALSE
CERTAINTY quadrant, entered via an advisory judge.

Defense-in-depth held: S6 fired for the dropped fire_1 (STOOD), S5
fired for car_1 (STOOD), pairwise+GT flagged the degradation, the
quadrant labeled it. Nothing was silently blessed — but the degradation
still shipped as the final answer.

**Proposed mitigations (pending approval):** (a) minority-advice caveat
in the composed runoff block ("4/5 of your own readings disagree with
this judge"); (b) substantiation gate — advice forwarded only if the
judge's reason cites a declared entity or state word.

**Claim:** advisory judges are an attack surface for the same
capitulation dynamics as reflection itself; authority must be earned
per-utterance (evidence-citing), not granted per-role.

---

## F6 · The instrument catches its own machinery
**Standing observation across F1, F2, F5**

Every degradation so far was caused by our own agentic apparatus
(reflection framing, prompt repair, judge advice) and every one was
caught by a different layer of the same apparatus (code checks, frozen
calibration reruns, GT quadrant). This mutual surveillance — no
component trusted, every component measured — is the design working,
and arguably the paper's central demonstration.

---

## F7 · The petition's maiden run (B_pool, live, 2026-07-22)
**Status:** rich partial success; three fixes shipped from one run

What happened, from the event ledger (ui_81f6c174):

1. **The fight:** assessment put BOTH drowning children in the threat
   slot (S5 ×2), reflection could not fix it (cap_reached) → petition
   fired, exactly per design, with both problems located in the
   previous answer.
2. **Partial resolution:** post-petition, child_2 left the threat list
   (S5 pressure halved: 2 violations → 1). child_1-as-threat persisted.
3. **The engulfing refusal:** the re-perception DID return the pool —
   with state `normal`. The baseline's designed escape (pool·engulfing
   when water contains a distress victim) is exactly what qwen declines
   to apply, twice now. With the pool declared "normal", the assessor
   still has no legal hazard, so the victim-as-threat pressure cannot
   fully resolve. Candidate next probe: does the fluid-convention
   wording in the perception prompt under-teach `engulfing`?
4. **Fresh eyes lost a victim:** the second look OMITTED child_2
   entirely (face-down, motionless, tiny distant bbox) and the
   lifeguard chair. The wholesale-replace merge accepted the deletion —
   a drowning child erased from the shared record by the repair
   machinery. FIXED: the NO-ERASURE RULE — petitions may add (two
   witnesses) but never delete; omitted originals are preserved and
   recorded as DISPUTES. Sunny's live run is the regression test.
5. **Weak reasons (Sunny's observation):** the reasons for the
   children-as-threats were generic. SHIPPED: weak_reason triggers —
   the talking-points check now FEEDS reflection (round 1 only, never
   rumination): uncited reasons get the offending text quoted plus the
   expectation ("name the declared state + the specific causal
   mechanism"). Mechanism-quality judging (R2) stays with the LLM
   rubric, on demand.
6. **UI shadow bug:** `for v in viols:` leaked and overwrote the
   verdict variable — the petition comparison printed `None·LNone`.
   Fixed + comparison now reads the verdict explicitly.

**Claims for the paper:** (a) re-perception under petition is not
monotone — a second look can LOSE true entities while gaining the
missing one; union-merge with dispute recording is the safe composition.
(b) A petition can be procedurally perfect and still fail on an
ontology-application gap (engulfing) — the trigger diagnosed the right
disease, but the cure needs the perceiver to know the convention.

---

## F8 · The engulfing refusal was a schema-language gap, not a model failure
**Date:** 2026-07-22 · **Scene:** B_pool (F7 follow-up) · **Status:** fixed in code (medium-bound derivation)

Sunny's diagnosis, verbatim in spirit: nobody says "water engulfing a
kid" — a kid drowns IN the pool. In natural language the hazard lives
inside the victim's verb; the schema wanted it as a state on the water.
`pool: normal` was qwen speaking correct English, twice, while the
ontology demanded a phrase outside its training distribution.

**Fix:** `derive_medium_hazards()` (perception.py) — if a living being
is in a medium-bound distress state (currently: drowning), the water
body hosting it is derived `engulfing · hazard_bearing` IN CODE, with
full provenance (`state_note` records what the model actually said and
which victim caused the override; a record note + `hazard_derived`
event + UI banner make it visible everywhere). Geometry disambiguates
between multiple water bodies; no water body perceived → nothing is
invented (distress with no visible hazard still legally yields empty
threats). Applied at the end of run_perception AND at the
run_assessment boundary (idempotent), so frozen records and
petition-merged records are covered alike. Precedent: enforce_kinds()
— the declared state decides; the model's claim is measured, not
obeyed.

**What it dissolves:** the S5 pressure behind every B_pool fight
(F1, F3, F7). With the pool derived hazardous the assessor finally has
a legal threat, so drowning children stop being drafted into the
threat slot to satisfy the quad ontology.

**Claim for the paper:** ontology terms must live inside the model's
natural language distribution, or be derived in code — never demanded
in the prompt. The failure mode is silent: the model answers the
distributionally-correct thing ("normal") and the pipeline reads it as
a perception error.

---

## F9 · The caption matcher fights the fluid convention
**Date:** 2026-07-22 · **Scene:** A_fire live (ui_390b7dd2) · **Status:** fixed (fluid-aware P5), model exonerated

The VLM's own perception followed the fluid convention exactly: house_1
·burning, no fire entity. Then P5's naive noun-match read "a house ON
FIRE" in the caption, found no 'fire' entity, and ticketed — so Loop 1
manufactured a redundant fire_1·burning. Downstream, S6 prosecuted the
model for leaving fire_1 out of threats. The model STOOD twice
(threats=[house_1] — exactly GT), paying for its correctness with
destabilized uncertainty: U 0.114 → 0.275 while GT error stayed 0.

One layer of the apparatus (the caption matcher) contradicted the
convention another layer teaches (attached fire is a STATE), a third
layer (S6) prosecuted the artifact, and only the subject's earned
stubbornness (F1's capitulation guard) kept the answer clean. When two
of our rules collide, the bug is in OUR rules, never the model.

**Fix:** fluid-convention-aware P5 — a caption's mention of an attached
medium is satisfied by an entity carrying the state: "fire"/"flames"/
"blaze" ← any burning/burnt entity; "water"/"flood" ← any flooded
entity. Kept deliberately strict everywhere else: free-burning phrasings
("brush fire", "wildfire") still demand a fire entity; smoke/dust/gas
are always diffuse and always owed; spill is still owed beside a leaking
producer (producer-and-medium rule). Rulebook P5 text updated to match
(same rules, two engines).

**Residual risk, accepted:** a caption saying just "fire" over a scene
with BOTH a burning structure AND a separate free fire will not ticket
the free fire. That gap belongs to the petition layer and the GT
harness, and is recorded here rather than papered over.

**Claim for the paper:** completeness checks against input text must be
ontology-aware, or they manufacture the very incoherence the pipeline
then spends model calls prosecuting. Cross-layer rule collisions are
detectable precisely because every layer writes to the same ledger.

---

## F10 · Check-pressure capitulation without a judge (the road promotion)
**Date:** 2026-07-22 · **Scene:** A_fire re-run (ui_34d8177e) · **Status:** fixed (S8 + richer geometry)

Post-F9 re-run: no fire ticket, no phantom fire_1 (F9 verified). But a
new shape appeared. P5 correctly demanded a road entity ("street" in
caption); scolded last run for out-of-vocab "paved", the model
overcorrected and declared road_1·burning — scene-primed state coercion
("dry" and "intact" were on the list). S6 then charged road_1 missing
from threats, and reflection resolved it the cheap way: the model
PROMOTED the road to threats with the reason "could be at risk if the
fire spreads" — a victim-shaped sentence in the threat slot. Run ended
"clean"; U tripled 0.067 → 0.225. No judge spoke: this is F5's
capitulation dynamic with pure check-pressure as the persuader.

The model's own reason text is the tell — it does not believe the
promotion it made. Repair authority flowed the wrong way: the evidence
indicted the upstream state, and the repair went downstream.

**Fixes shipped:**
1. **S8 `threat_reason_victim_shaped`** (code check, deterministic,
   direction-sensitive: "is/could be at risk" matches, "puts X at risk"
   never does) + rulebook chunk with both legal exits (leave threats, or
   stand by the state and name what it harms; disputing the state is
   recorded, not punished) + talking-points card line. S8 surviving
   reflection is PETITIONABLE — the contradiction indicts the perception
   artifact, so the petition path opens, per the loop principle.
2. **Richer overlap hints** — the run also showed every geometry pair
   reading identically as "overlap, 0px" (whole-street road box, large
   house boxes). Overlap hints now carry containment (% of the smaller
   box covered) and center distance; edge-gap stays for adjacency.

**Claims:** (a) advisory judges are not required for capitulation —
any pressure channel (checks included) can extract a schema-legal,
causally-absurd answer; defenses must read the STORY, not just the
slots. (b) Repair prompts have side effects on later runs of the same
scene (the "paved" scolding produced "burning") — state coercion is
F2's lesson recurring at the state level.

**F10 addendum (same day):** the P3 repair instruction itself was an
accelerant. Its old rendering chained "...matches what you see:
hazard-bearing: burning, ..." — a colon structure that binds "what you
see" to the hazard family, with `burning` as the first legal word
offered, in a fire-saturated scene. Sunny misparsed it the same way on
first read; a 7B model plausibly did too. Fixed three ways, all
presentational (no rule change, prompt neutrality intact): (1) NORMAL
family now leads the word list — positional bias now pulls toward the
statistically honest prior ("most entities in most scenes are normal",
said explicitly); (2) the colon chain is broken; (3) one closing line
names the failure generically: "pick the word that matches THIS
entity's own condition, not the scene's overall situation." Plus
`paved` admitted to EXTRA_NORMAL_STATES — the model's honest first
answer for the road no longer draws a ticket at all, so the coercion
round never happens. Claim: repair-prompt MICRO-STRUCTURE (ordering,
punctuation) is an intervention surface with measurable side effects,
same law as F2 at a finer grain.

---

## F11 · "Active" is what people say about a spill (C_tanker, ui_529ce417)
**Date:** 2026-07-22 · **Status:** both fixes shipped

Two findings from one run.

**1 · The unknown-kind blind spot.** P5 (post-F9 strictness) correctly
demanded the spill entity beside the leaking tanker. The model added
`spill · "active"` — natural English — P3 ticketed it, the model STOOD,
and the state landed kind=unknown. Consequence: a DECLARED FUEL SPILL
silently vanished from every Stage-2 check (S6 guards hazard_bearing
only); threats shipped without spill_1 and nothing fired. Fix:
LABEL-AWARE state synonyms — a global map can't hear "active" because
it means a different canonical state per medium: (spill, active) →
seeping; (fire, active) → spreading; (smoke, active) → billowing;
(water, active) → rising; plus "pooling" → seeping. Applied in the
record AND in P3 (the honest word no longer draws a ticket, so the
stand-off never happens). Same law as F8, one level finer.

**2 · The pairwise judge was asked an undefined question.** Reflection's
round changed only prose; the auto-pairwise gate keyed on `changed`
(any text diff), so llama was forced to rank two answers whose DECISION
layers were identical — and it confabulated a preference ("POST is more
accurate because it correctly identifies..."). Noise wearing an F4
costume. Fix: SUBSTANTIVE-CHANGE GATE — auto-pairwise convenes only
when the decision layer moved (scenario, type FAMILY, severity BUCKET,
threat set, at-risk set+kinds — the same folds the probe machinery
uses); prose-only changes log `pairwise_skipped` and the judge card
says so. Within-bucket level wiggles and type re-wordings never summon
a judge (that is the U-machinery's jurisdiction). This run's verdict
is EXCLUDED from the F4 table.

Also observed, third instance: U rose through a "clean" reflection
(0.143 → 0.2; type and bucket splits appeared after a prose-only
change) — reflection perturbs stability even when it changes nothing
substantive. Pattern now has three sightings (B_pool F7, A_fire F10,
here); candidate name for the paper: "reflection jitter".

**Claim:** judge questions must be well-posed before judge answers are
data — a forced choice between equivalent options measures the judge's
bias, not the answer's quality (and belongs in F4 only as a bias probe,
never as a quality verdict).

---

## F12 · A whole petition ran on one missing synonym (D_aerial, ui_c7b362ef)
**Date:** 2026-07-22 · **Status:** fixed (compound spill synonyms); apparatus behaved correctly throughout

The chain: P5 demanded a spill (caption "chemicals"). The model answered
`chemical_spill` — MORE specific than our noun — but the synonym map
only knew "chemical". So: label fell to `other` (escape hatch), DINO
had no noun to ground (SAM fallback box), P3 ticketed the state, the
caption ticket for 'spill' STOOD through cap_reached... and the
petition fired, exactly per design. The second look omitted the entity
entirely; NO-ERASURE kept other_1·spreading in the record as a DISPUTE,
nothing merged, no cascade. Meanwhile the assessment layer did its job:
S6 caught other_1 missing from threats, reflection added it (plus
ambulance·proximity from the new rich geometry hints), runoff convened
at U 0.222 and sided with the majority reading (F4 data point: no
severity-minimizing this time).

**The lesson:** every layer executed its charter correctly, and the run
still burned ~20 model calls prosecuting a vocabulary gap. Upstream
name resolution is the cheapest layer in the whole stack — and the only
one that could have made the entire episode unnecessary. Fixed:
chemical_spill / oil_spill / fuel_spill / spillage / hazmat_spill →
spill.

**Also:** 4th sighting of reflection jitter (U 0.222 → 0.275 through a
"clean" round). The pattern is now consistent enough to measure
systematically after the calibration set completes.

**Claim:** in a layered repair architecture, the cost of a gap is paid
at the most expensive layer that can compensate for it, not the
cheapest one that could have prevented it. Vocabulary completeness is
therefore a first-class reliability concern, not housekeeping.

---

## F13 · The locked-room ticket, and five humans in a three-human scene
**Date:** 2026-07-22 · **Scene:** E_collapse (ui_20fb0754) · **Status:** both fixed · **Taxonomy: C + B**

Assessment was flawless (Yes · Structural Collapse · L8, S6 pulled
dust_1 into threats, person_1·distress, zero jitter — first run with
U flat). Both defects were upstream, and both were ours.

**1 · The locked room (Cat C).** The model labeled a police car
'infrastructure' (family name) while its own description said "police
car with flashing lights". Our P1 ticket quoted ONLY the infrastructure
family's members and closed with "if none fits, use 'other'" — a menu
without the answer, twice, so the model lawfully fell to other_1. The
correction was a locked room. FIX: the P1 template now quotes the
entity's OWN description back and permits the full vocabulary ("ANY
family, not only '{raw_label}'"). Evidence-first, never coaching — the
description is the model's own words.

**2 · Duplicate humans (Cat B, machinery-induced).** The model listed
the two officers as person_2/person_3 (honest descriptions: "police
officer behind caution tape"). P5's family-lenient match treats person
and responder as different families, so the 'police_officer' caption
ticket fired and the model ADDED both officers again: five humans in a
three-human scene (Sunny confirmed ground truth: one man trapped, two
officers). Phantom lives corrupt geometry pairs, membership votes, and
Stage 4 consequence weighting. FIX: P6 `duplicate_entity` — two
same-life-group entities (person/responder = one human group; animal
separate) with IoU ≥ 0.8 draw a ticket quoting both descriptions; the
MODEL resolves (merge under the better label, or keep both and
disambiguate). Calibrated on this run: true duplicates ≈ 0.94 IoU,
the two real adjacent officers ≈ 0.1.

**Claim:** correction channels need the same design care as first-pass
prompts — a repair menu scoped to the wrong category converts a small
error into a permanent one, and a completeness check blind to identity
mints phantom people. Both fixes use only the model's own prior words
as evidence.

---

## F14 · Petition routing: send the complaint to the stage that made the mistake
**Date:** 2026-07-22 · **From:** E_collapse (ui_e45e9956) · **Status:** built

That run's petition re-looked at the image — but the image was read
correctly. The mistake was the sorting: the trapped man was put in the
threat list. An image look can't fix a sorting mistake, so it came back
empty ("unresolved") after a full re-perception.

Fix, per Sunny's routing idea (and the loop principle — repair
authority follows the evidence):

  entity list might be wrong  → stage 1: re-look at the IMAGE
    (caption names something nobody found, or the verdict wants a
     danger source and the record declares none — the B_pool shape)

  facts fine, sorting wrong   → stage 2: re-ask the QUESTION once
    (legal hazards already declared; a fresh clean ask, previous
     answer shown, problems quoted with their rules, "return it
     unchanged if you stand by it"; one model call, cap 1, the
     record is never touched)

On the E_collapse shape this costs 1 model call instead of a full
re-perception cascade, and aims at the actual patient. UI shows
"PETITION → SAME STAGE (question re-asked fresh)" with the perception
panels left alone. Also added: the WHERE-ARE-WE progress strip (per
Sunny) — every pipeline step as a chip: ✓ done, ● now (pulsing, with a
plain line of what's happening), ○ not started.

**F13 addendum (E_collapse re-run ui_bcc80931):** the P6 ticket fired
correctly in round 2 — but the model kept both copies and the round cap
ran out, so the five-humans record shipped anyway. The model gets asked
first; a duplicate that STOOD is now resolved IN CODE at assembly:
same-group entities with IoU >= 0.8 merge, the more specific label wins
(responder beats person), the dropped entry is written into notes and a
`duplicate_merged` event ("nothing about the scene was lost").
Same precedent as enforce_kinds: identity at near-total overlap is a
geometric fact, not a perception judgment. Also explains Sunny's
overlay confusion: the officer boxes were EXACTLY underneath the
person boxes, so only one label was visible per human.

---

## STAGE 2 — FROZEN (2026-07-23)

Stage 2 (merged scene assessment) is closed. Six scenes calibrated live;
findings F1–F14 each fixed and regression-tested; F_park_control passed
the silence test (a safe scene produces nothing). Central result:
**most defects were ours, not the model's — "fixing the interview, not
the witness."** The model's true deficits: capitulates under
authoritative pressure; unstable second looks; flat 0.95 self-confidence.

From here, no behavioral changes to Stage 2. Next work is INFRASTRUCTURE
(LangGraph control refactor, built alongside and proven output-identical)
and then Stage 4 (recommendations / causal quads). Suggested git tag:
`git tag stage2-frozen`.

---

## INFRA · LangGraph control + RAG shadow (2026-07-23)

**LangGraph control (graph_live.py), built alongside — proven identical.**
The Stage-2 + petition control (assess → router → re-look / re-ask →
cascade) is now expressible as a LangGraph StateGraph. The petition
router is a real 3-way conditional edge; the cascade is a back-edge.
Selected with `AGENTIC_CONTROL=langgraph` (default `python`). The two
paths are proven byte-identical on record, result, petitioned AND the
event stream, across all three router branches, given the same mocked
model answers (test_graph_live.py). "Exact same output" is hermetic —
live runs can't match (probes are temp 0.7 by design). The two inner
loops (repair, reflection) stay inside their functions for now;
decomposing them into self-edges is a clean v2 guarded by the same test.

**Retrieval switch + RAG shadow (retrieval.py).** `AGENTIC_RETRIEVAL` =
rulebook (default, exact-key) | rag (RAG top-1) | both (exact-key
authoritative, RAG recorded). Default 'rulebook' is byte-identical to
the old lookup, so the pipeline and the LangGraph equivalence are
unchanged. Reflection's rule-quoting now routes through the switch.

**Agreement report — exact-key vs RAG top-1 (rag_agreement_report.txt).**
On this cloud box HuggingFace is blocked (403), so the number shown is
the **keyword-fallback** engine: **11/18 = 61%**. The 7 mismatches are
exactly the near-neighbour confusions we predicted on a small rulebook
— e.g. "missed a disaster" (S2) and "threat not hazardous" (S5) both
pulled toward *caption_entity_missing*; the geometry/at-risk trio bled
together. This is precisely why exact-key stays the pipeline default:
when you hold the key, semantic search can only match it or get it
wrong. Run `python -m agentic.retrieval` locally (HF available) for the
real BAAI/bge-small number — expected higher than keyword, but the
lesson stands. RAG's real home is Stage 4 (semantic checks, no key).

---

## F15 · Stage 4 scoring metrics — audit before building (rules mostly sound, numbers misleading)
**Date:** 2026-07-24 · **Scene:** N/A (code audit of Arm A metrics we import) · **Status:** open — corrected Arm B layer to be built in Phase 1b

Before wiring Stage 4 to Arm A's conformance + alignment scoring, audited
`check_graph_rule_conformance`, the validity formula, `assess_pre_internal_alignment`,
and `compare_graphs{,_soft,_topological}`. The 20 conformance rules and the
internal cross-checks are coherent and mostly correct. The SCORING NUMBERS
have four real defects that make Arm A's trust read persistently, misleadingly
low (Sunny's observation: Arm A trust is always low because of all the violations):

1. **Saturation.** conformance_validity = 1 − min(1, violations/edges). Once
   violations ≥ edges it pins to the floor — a graph with 5 violations and one
   with 40 (on 5 edges) score identically. Throws away the magnitude that IS the
   signal. (VLM breaks tens of rules, so this floors constantly.)
2. **Wrong denominator.** Numerator includes node-level + whole-graph violations;
   denominator is EDGE count only. Same node defect scores differently by edge
   density. Principled denominator = nodes+edges (or per-family).
3. **Double-counting.** One conceptual error trips 2–3 rules (drowning entity
   flagged hazardous → hazard_flag_state_mismatch + hazardous_and_at_risk
   [+ distress_state_on_non_living if non-living]). Inflates numerator, accelerates
   saturation.
4. **Severity-blind.** Flat head-count weights cosmetic rules (node_budget_exceeded,
   redundant_instancing = no_effect) the same as fabrications (unresolved_endpoint).
   Ignores Arm A's own CONSEQUENCE weighting.
   (A/B asymmetry — A floors at 0.5, B at 0.0 — is INTENTIONAL + documented, not a
   defect, but the two numbers are on different scales; must be labeled.)

**Internal alignment** (assess_pre_internal_alignment, main.py:1373) is real and
mostly enforced, but THREE prompt rules are NOT enforced in code: Rule 4's
"reason ids in related AND quad" is relaxed to OR (line 1700); the "(verb,target)
action-collapse" rule + self-check (i) never inspect `action`/`expected_consequence`;
"threatens as last resort" (g) unenforced. Plus a presumed-id inconsistency
(reason-coverage demands the presumed token verbatim; related-coverage excuses it)
→ false-positive risk.

**A-vs-B (declared Graph B vs structured Graph A):** the precision/recall pair is
CLEAN — a_fidelity = |A∩B|/|A| (how much of the recs' graph B backs up),
b_coverage = |A∩B|/|B| (how much of B's declared graph the recs reproduce),
B the yardstick. Good, keep it. BUT the three "structural" agreement tiers use
THREE different math definitions — strict = set-Jaccard, soft = Dice, topological =
multiset-Jaccard — so they are NOT comparable (soft reads higher by formula alone)
and don't form a monotone loose→tight ladder (topological can score < strict on
duplicate edges). Two different metrics are both named "topological." Pick ONE
structural definition; don't threshold the three against each other.

**Decision (Sunny, 2026-07-24):** IRON RULE 1 stands — never edit main.py; import
only. So we cannot fix Arm A's rules/metric. Instead, Phase 1b builds a CORRECTED
Arm B trust layer that (a) imports the raw violation LIST from
check_graph_rule_conformance, then dedupes the double-counts (group by entity/defect),
severity-weights (drop no_effect rules from the trust penalty, weight by Arm A's
own CONSEQUENCE), and normalizes by nodes+edges without hard saturation; (b) enforces
the three prompt-only internal rules as real Arm B checks; (c) uses one structural
definition. Arm A's raw frozen numbers are STILL recorded alongside, for arm
comparability. The trust panel shows the VIOLATION BREAKDOWN BY CATEGORY (which rules,
how many, severity) — the honest "where + why we can't trust the VLM" — not a single
saturated number.

**Claim for the paper:** a naive violations/edges validity saturates and hides the
model's failure profile; the informative signal is the *distribution* of rule breaks
by category and severity, not a scalar.

---

## F24 · One law, two witnesses — the recommendation card
**Date:** 2026-07-28 · **Scene:** D_aerial_spill (round 3) · **Status:** built, 520 tests

A recommendation card carries three surfaces. The ACTION is written
first; the prose REASON and the structured QUAD are both written
afterwards to explain it. Two defects, both ours:

**1 — one witness under law, one witness free (category C).** The quad's
threat had to come from the `threats:` line, its state from the declared
states, its verb from the closed effect list. The prose reason had NONE
of those constraints. So the model wrote a free-form subject, reached the
quad, found that subject illegal, and swapped it — and the alignment
check scored the swap as the model's defect. On D_aerial's card 1 the
reason blames `tanker_truck_1` and the quad blames `spill_1`, which is
exactly that forced swap.

**2 — the identity check could not see a reversal (category F).** The
old check compared SETS OF IDS. `A --may_harm--> B` and
`B --may_harm--> A` share the same ids, so a reversed causal direction —
the error CEE+ exists to find — passed clean. Separately, the prompt's
instruction that an action must name its entities by object_id was read
by nothing at all, so the model ignored it for free ("Secure the tanker
truck").

**Fix.** The reason now carries the quad's four rules, and the quad is
told it is that same sentence with its slots filled. 22 rules, tagged
and routed into the two reports that already existed — no third score,
so the trust weights are untouched:

- **conformance (16)** — one surface against the rules: the action, the
  prose reason, remaining_risk, rank.
- **internal alignment (7)** — surface against surface, at the level of
  ROLES not ids: `subject_mismatch`, `object_mismatch`, and does each
  explanation cover what the action operates on.

Three amnesty rules record at severity 0 instead of charging, from ONE
shared predicate: a victim with no declared hazard, a hazard with no
declared victim, and an entity the scene put on both lines. In each the
model had no legal word available — our constraint, not its error. The
first cut got this wrong in the characteristic way: it forgave the prose
and billed the structure for the identical situation.

`action_mode` is recorded per card (hazard-directed / victim-directed /
mixed / unattributed), read off which quad slot the action's ids land in
— no keyword list, no English to parse. It is not a defect; it is what
the intervention gate needs to know which recommendations it can test at
all. Victim-directed actions need their own intervention family.

An advisory JUDGE runs beside it, display-only, never scored — so it is
safe to leave on during calibration. The rule tier catches INVERTED; the
judge tier catches HOLLOW (every identity rule passes, and the
explanation still says nothing causal). Both explanations answer the
same three questions, because both explain the action.

Every card's findings now render UNDER that card. Before this the checks
fired in one panel and the evidence sat in another.

**Flowchart:** Stage 4 gains a CARD CHECK box between `evals` and
`trust`, splitting into two arrows that rejoin the existing conformance
and internal-alignment boxes, plus a dashed advisory arrow to a JUDGE
box that touches no score.

**Open:** the rules score ONE sample (the temp-0 answer). The five probe
recommendation sets are discarded after reduction; running the same rules
over them would separate "breaks this rule sometimes" from "broke this
rule" at no model cost. Deferred by Sunny until conformance/alignment/
judge have been seen on live scenes.

---

## F25 · The quad the model got right, that we deleted
**Date:** 2026-07-28 · **Scene:** A_fire (round 4) · **Status:** fixed, 529 tests

All three recommendation cards showed `N/A · N/A --N/A--> []`. The model
had written every quad CORRECTLY — as an arrow string:

    "house_1 -> burning -> may_harm -> person_1, dog_1"

The frozen normalizer takes a non-dict quad to `{}`, which becomes the
all-"N/A" placeholder. Graph A came out empty, alignment and trust were
meaningless, and the new card checks charged the model for a quad we had
deleted. **`parse_notes` was empty.** A total loss of the one thing
Stage 4 exists to measure left no trace anywhere.

Three defects, all ours:

**1 — the prompt made a JSON field read like prose (category B).** F24
had rewritten the quad's instruction as "the SAME sentence as the reason,
with its slots filled in". The model wrote a sentence. Restored to "a
JSON OBJECT with exactly the four keys below — never a sentence, never an
arrow string", keeping the restatement meaning.

**2 — a boundary that ate raw model output failed silently (category F).**
`coerce_quad` now recovers `->`, `-->`, `→`, `·` and `--effect-->`
dialects before the frozen normalizer sees them, with a note. More
important is the GENERAL guard beside it: any recommendation that ARRIVED
with a quad and LEFT without one is noted as `LOST_IN_PARSE` whatever its
shape. That is the guard that catches the NEXT shape, not only this one.

**3 — we asked for plain English and then demanded a token (F25b).** The
reason is specified as "plain English", so the model wrote "it may harm
person_1". The effect matcher required `may_harm`, and charged severity 2
on all three cards for obeying the instruction. The underscore is our
serialisation, not a word the model owes us in prose; both spellings are
now accepted.

**After the fix, on the same raw answer:** card 1 clean and
victim-directed; cards 2 and 3 unattributed because their actions name no
object_id ("Alert emergency services about the burning house"), and card
2's reason names no legal mechanism at all. Three real findings that the
`N/A` had buried. Card 2's quad is also a genuine self-loop —
`house_1 --worsens--> house_1` with two at-risk entities declared.

**The pattern holds.** Of everything the cards reported on this run,
almost all was ours; what survived the fixes is small, specific, and
real.

---

## F26 · The judge rubric that scored the right answer as wrong
**Date:** 2026-07-28 · **Scene:** A_fire (round 5) · **Status:** fixed, 531 tests

A_fire produced two clean, well-grounded rescues:

    Rescue person_1 — Because house_1 is burning, it may_harm person_1.
    Rescue dog_1    — Because house_1 is burning, it may_harm dog_1.

The advisory judge flagged **all four surfaces** as restatements and all
four as failing necessity. Sunny caught it: "The judges are wrong. The
reasons and quads explain why the action is necessary for both recs."
Both errors were in the RUBRIC, not the judge.

**1 — the gloss defined the right answer as a failure.** Q1 said "an
explanation that names a danger and stops has not said why THIS action
follows from it". For a rescue, naming the danger IS the complete cause.
Rewritten to say so explicitly. The value names also moved from
`cause`/`restatement` to `causal`/`circular`: the card's OWN spec uses
"restatement" for the quad restating the reason — which is required — so
asking about restating the ACTION in the same word collided with it.

**2 — same answer key, opposite meaning.** Q3 was phrased per
action_mode, and the two phrasings were INVERSES:

    hazard-directed: "would this action become unnecessary?"  no = UNGROUNDED
    victim-directed: "would the protection still be needed?"  no = GROUNDED

The flag counter treated `necessity == "no"` as concerning in both, so
the grounded answer was counted as a defect. Card 1's two verdicts even
carried notes asserting OPPOSITE things and both scored "no". Asked in
ONE direction — "if the danger were removed, would this action still be
needed?" — it works for either mode, and "yes" is the concerning answer
in both. The mode split was over-engineering, argued for in F24 on the
grounds that a mode-blind question "measures nothing". That was wrong.

**Wider point.** This is the same defect class as F24 and F25, now in the
judge tier: our own instrument punishing the model for obeying our own
instruction. It is also the argument FOR the judge shipping
advisory-only — a rubric bug this size would otherwise have moved the
trust weights during calibration.

**Standing on the run itself:** trust 0.801, picks unanimous,
a_fidelity 1.0, measured uncertainty 0.179, Graph B uncertainty 0.05,
zero alignment failures. The one real card defect was
`remaining_risk = "[car_1, proximity]"` on both cards — a role word where
a state belongs, and duplicated across the set. Both caught.

**Open (new):** both recommendations are `victim_directed`, so NOTHING in
this run is testable by hazard suppression, and Graph B's
`house_1 may_spread_to car_1` is acted on by no recommendation
(b_coverage 0.667). "Protects victims, never addresses the cause" is a
pathology shape `action_mode` now makes visible — but nothing reports it
yet. Candidate set-level signal.

---

## F27 · The instruction that was stricter than the rule it described
**Date:** 2026-07-28 · **Scene:** A_fire · **Status:** fixed, 534 tests

Both recommendations came back victim-directed — two rescues, nothing
addressing the fire. Nothing in the run was testable by hazard
suppression, and Graph B's own `house_1 may_spread_to car_1` was acted on
by no recommendation (b_coverage 0.667).

**It was not the model.** Every one of the 15 prior A_fire runs in the
archive carried a hazard-directed recommendation ("Deploy firefighters to
extinguish house_1"). Only the two runs AFTER the F24 prompt edit lost
it.

The action clause read:

    Name the entities it operates on by their object_id, exactly as
    listed above — never by a prose description of them.

Written to catch "Secure the tanker truck". It also reads as an absolute
ban on any noun that is not an object_id — and responders, teams,
vehicles and equipment are never in a Stage-1 entity list, because
Stage 1 perceives the SCENE, not the response. So the safest card the
model could write was one whose every noun was on the list:
`Rescue person_1` / `Rescue dog_1`. It also shrank from 3 recommendations
to 2. It optimised for passing our constraints rather than for responding
to the emergency.

**Nothing in code ever forbade it.** Sunny pushed back on the claim that
the action was illegal, and he was right: the checks read the scene ids,
classify the action `hazard_directed`, and never look at the other words.
"Deploy firefighters to extinguish house_1" passes with zero findings.
The INSTRUCTION was stricter than the rule it described, and the model
believes the instruction, not the checker.

**Fix (prompt only, no code change).** The clause now says a scene entity
the action acts on must be named by its object_id, and that the list
describes the scene rather than the response — so responders, teams,
vehicles, equipment and services may be named in ordinary words. Neutral
across disaster types by construction; a test asserts the clause carries
no domain noun at all (iron rule 5).

**Caveat, stated:** n=2 after the change against 15 before. A clean break,
not a proof — the model could be sampling differently. The wording fix is
right either way, since an instruction stricter than its own rule is a
defect on its own terms.

**Pattern.** Third induced error this week, all the same shape: a real
defect fixed with a rule wider than the defect. F25 (a JSON field
described as prose), F26 (a rubric that scored the correct answer wrong),
F27 (a ban wider than the check). The finding holds and sharpens — most
defects are ours, and the recurring mechanism is over-correction.

---

## F28 · The judge was missing a definition, not a brain
**Date:** 2026-07-28 · **Scene:** A_fire · **Status:** wired, 8/8, 535 tests

The card judge scored 0 for 5 on a clean run (F26), and its replacement — a
six-step "walk the causal chain" rubric — scored 0 for 4 on the one case the
tier exists for: a card where every code check passes and the action makes
nobody safer. Two rubrics, four rewrites, opposite biases, no discrimination
either way. The working conclusion was that llama3.1:8b could not do the task.

Sunny rejected that: "Something smells fishy. A modern llm cannot reason?"

He was right. Asked why a hollow card was aligned, the judge answered in plain
prose:

    "Photographing house_1 directly addresses the immediate concern of
     DOCUMENTING the hazard that poses a risk to person_1"

The reasoning is fluent and coherent. It was applying "the action is about the
same hazard" as its test — because we never told it what causally aligned
MEANS. Not a reasoning failure, a CRITERION failure.

One sentence fixed it:

    An explanation is causally aligned only if carrying out the action would
    REDUCE the harm that explanation describes.

Same model, same scene: 0/8 -> 7/8 on a fixed discrimination set. Everything
built before it — three calls, six steps, forced JSON, five rubric revisions —
was scaffolding around a missing definition.

**Two further findings, both load-bearing.**

*The forced JSON was suppressing the reasoning.* `response_format:
json_object` makes the model emit `{` from the first token, so the verdict
lands before any thinking happens and the "steps" are just keys filled left to
right. Removed; the judge answers in prose and signs off with two parseable
lines.

*Every clarification we added FLIPPED the model wholesale* rather than
sharpening the boundary. criterion alone 7/8; + a notation gloss 7/8 with the
error in a different cell; + one more gloss 4/8 with every verdict negative.
Deterministic each time (5/5 identical at temp 0), so these are stable
positions, not noise. The prompt is now frozen and small.

**Five probes, majority vote (Sunny).** At temperature 0 a wrong verdict is
wrong 5/5 and looks confident, and the boundary cases are exactly where one
sample says least. Asking five times at probe temperature and taking the
majority took the frozen prompt from 7/8 to **8/8** — the one stably-wrong cell
came back aligned 4/5. The VOTE SPLIT is displayed beside every verdict: a 3/2
is the judge saying it cannot see the boundary, which is worth more than
whichever side it landed on.

**The real asset is the test set, not the prompt.** Four cards, all code-clean,
two hollow and two good. Before it existed, prompts were judged by reading a
few verdicts and forming an impression — which is how a 0-for-5 judge shipped
and how one of its wrong verdicts got called "genuinely useful". It wants
growing: an inverted card, a partially-aligned one, a card whose action helps a
different victim than the one named.

**Pattern.** Fourth instrument defect this week and the same shape as F25/F26/
F27: our own machinery punishing the model for something we failed to specify.
The wider finding holds — most defects are ours — and the mechanism is now
named twice over: over-correction, and unstated criteria.

---

## F29 · The findings that belonged to no card
**Date:** 2026-07-28 · **Scene:** D_aerial_spill · **Status:** built, 559 tests

F24 gave every recommendation a card footer carrying its own findings. That
works for anything wrong with ONE card. It has nothing to say about findings
that belong to the COLLECTION:

    at-risk dog_1 is not addressed by any recommendation
        nothing is wrong with card 1, nothing is wrong with card 2 — the dog
        is simply absent from both
    rec 2 has the same quad as rec 1
        which card is at fault? neither alone
    every card ranked #1
        no card is wrong; together they do not triage

**Seven such rules were firing, split across two eval functions, and five
rendered nowhere.** Only `rank_not_a_triage` and `remaining_risk_duplicated`
reached the small set box F24 added; the at-risk coverage rules, the duplicate
quad, the action collapse and the older duplicate-remaining-risk rule were
computed, scored, and then dropped on the floor.

D_aerial round 4 shows the cost. Both hazmat workers went unaddressed —
severity 2 each — and neither appeared on screen. The abstract form of the same
fact, `b_coverage 0.00`, sat in another panel with `ab_alignment` WITHHELD by
the trust gate. The readable version of the finding existed and was discarded;
the unreadable version was suppressed as untrustworthy. The run displayed
trust 0.659 and said nothing about two people.

**Fix.** Every finding now carries `signal` (conformance | internal_alignment)
and `level` (card | set) — the same two fields card findings already had. The
three pairwise rules are `level="set"`. The two at-risk rules are additionally
`signal="conformance"`, because "every at-risk entity must be acted on" is a
LAW about the set, not a comparison between two things; they had been filed
under alignment since before that split existed. They are still SCORED where
they always were, so run-to-run numbers stay comparable — only the tag moved,
and a test pins the score against the tagging.

`set_report()` aggregates them into two groups, kept apart because they map to
different pathologies at S5: COVERAGE is under-response, PAIRWISE is padding —
count produced in place of content. The panel always renders, unlike a card
footer: a missing footer means "this card is clean", but a missing panel would
be ambiguous between "no cross-card problems" and "not rendered", and the
absence of duplication is a real positive signal.

**A third gap, and the one that matters most.** Nothing computed what the set
ACTS ON. D_aerial produced two `unattributed` recommendations, which means
NOTHING IN THAT RUN WAS TESTABLE BY HAZARD SUPPRESSION — the single most
important sentence about it, and it appeared nowhere. The mode rollup now says
it in words. It is not a violation; it is what the intervention gate (S6) needs
in order to know which recommendations it can test at all.

**Flowchart:** no pipeline change — one aggregation plus display. The CARD
CHECK box edit owed from F24 still stands.

---

## F45 — a_fidelity called perfect agreement "nothing in common", and the
## comparison could not hear the model's own synonyms

**Categories: F (metric defect) + A (language gap).** Round-2 D_aerial.

**What Sunny saw.** The run was good. The A-vs-B panel said:

```
a_fidelity  0.00        b_coverage  0.00
```

0.00 means "the advice shares nothing with what the model independently
believes". That is not what happened. What happened was:

```
Graph A (from the recommendation quads)   Graph B (asked independently)
  spill_1 -blocks_access_to-> fire_truck_1  spill_1 -may_harm-> hazmat_worker_1
  spill_1 -blocks_access_to-> ambulance_1   spill_1 -may_harm-> hazmat_worker_2
  spill_1 -blocks_access_to-> police_car_1  tanker_truck_1 -may_spread_to-> spill_1
```

The model agreed COMPLETELY about what the danger was — the spill, both times,
`hazards 1.00`. It disagreed about who the spill endangers: the advice
protected three vehicles, its own beliefs named the two people standing in it.
That is a precise, serious, fixable defect. `0.00` is the same number two
graphs with nothing whatever in common would produce, so the panel destroyed
the distinction it existed to draw.

**Two separate causes.**

*Cause 1 — the match was too strict (F).* A match meant the WHOLE edge:
source, effect AND target. So the effect word had a veto. And the effect word
is the least stable part of the claim: `exposes`, `may_harm` and
`may_spread_to` all came out of the same model, for the same tanker, on the
same scene. Counting those as three disagreements measured VOCABULARY, not
grounding. On D_aerial `blocks_access_to` vs `may_harm` alone was enough to
zero a comparison that agreed perfectly on the hazard.

*Cause 2 — the ids were compared as strings (A).* Graph B has repeatedly
written `chemical_spill_1` for the entity the scene calls `spill_1`, and
`chemical_worker_1` for `hazmat_worker_1`. The closed vocabulary ALREADY knows
`chemical_spill -> spill` — Stage 1 uses that very map to name entities. The
comparison had its own, blinder rule, so the same word arrived canonicalised
on one side and raw on the other, and one agreement was scored as a
fabrication AND an omission at the same time. This is the F8/F11/F12 language
gap again, in a new place: our machinery could not hear the model's own
synonym.

**Fix, part 1 — a_fidelity is built from two overlaps, effect ignored.**

```
a_fidelity = mean( same hazards , same victims )

  same hazards   1.00     it agreed about what the danger was
+ same victims   0.00     it pointed that danger at the wrong people
= a_fidelity     0.50     half right, and you can see WHICH half
```

D_aerial round 2 went from `0.00 / 0.00` to `0.625 / 0.666`. Nothing was
thrown away: the whole-edge pair is still computed and returned as
`a_fidelity_strict` / `b_coverage_strict`, and the panel has a toggle (default
IGNORED, one click to counted), so every number quoted from every earlier run
stays reproducible. Where the effect word genuinely changes what a responder
would DO, the graph judge's second question asks about it directly — that is
the right instrument, because it is a judgment and not a string comparison.

**The blind spot, stated rather than hidden.** A mean of two SETS does not
check the WIRING. Two graphs naming the same hazards and the same victims but
connecting them to each other differently score 1.00 while agreeing on no
single claim. `pairs` (who threatens whom, effect still ignored) is the number
that catches it; it is computed, it is inside the toggle, and when the wires
ARE crossed the panel says so in a warning. It is deliberately NOT folded into
a_fidelity, because folding it in re-introduces the 0.00 this change removes.

**Fix, part 2 — the id matcher is a three-rung ladder.** First rung that lands
wins, so a loose match can never displace an exact one:

```
1  verbatim    the stem IS a scene entity's label or state
               chemical_spill_1 -> spill_1   (state was `chemical_spill`)
2  synonym     both canonicalise to the same vocabulary word, through
               vocabulary.LABEL_SYNONYMS — the map Stage 1 already uses
3  head noun   the last word matches: chemical_worker <-> hazmat_worker
```

Ambiguity rules: one candidate is an alias; several candidates of DIFFERENT
labels is no alias, because a wrong merge is worse than a visible mismatch;
several candidates of the SAME label is the two-workers case, and there the
number the model itself attached decides — `chemical_worker_2` is numbering the
series we numbered. Refusing there was the expensive option, since it made one
claim count as a fabrication and an omission at once. Which rung fired is
recorded and printed, so `(head noun, by number)` reads as the weaker claim it
is and a loose merge is never silent.

**A bug found while building it.** `canonicalize_label` sends everything it
does not recognise to `other`. Using its output raw put every unrated word —
every free-text state — into ONE bucket, and rung 2 then found several
unrelated candidates under it and merged nothing while blocking rung 3. Rung 2
now requires a real vocabulary hit. Caught by a test, not by a run.

**TRUST IS UNCHANGED, deliberately.** The `ab_alignment` trust contributor
reads `structural`, which still comes from Arm A's frozen comparator on whole
edges. So the trust weighting stays comparable with every run to date; only
what a reader is shown, and how it reads, has changed. Pinned by a test.

**Status.** 652 tests pass. Old runs replay with their stored numbers, because
the UI is a pure function of the event stream — the new numbers appear on the
next live run of each scene.

**Flowchart:** no pipeline change — a metric definition plus display. The CARD
CHECK box edit owed from F24 still stands.
