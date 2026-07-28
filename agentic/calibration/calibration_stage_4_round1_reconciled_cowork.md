# Stage 4 calibration — round 1: reconciled findings and final proposal

**Author:** Claude (Cowork). **Revision 2** — updated after Claude Code revised
its own analysis in response to revision 1.

**Inputs:**
`calibration_stage_4_round1_claudecode.md` (rev 2, F-series) ·
`calibration_stage_4_round1_cowork.md` (rev 1, F16–F21).

**Status:** proposal only. Nothing is built. Two items need Sunny's explicit
ratification before any code is written — flagged in §7.

Runs: A `ui_4b6355e1` · B `ui_e50abbb5` · C `ui_1430cc13` · D `ui_e06a9547` ·
E `ui_6b2d9163` · F `ui_c4c41d47`.

---

# 1. What changed since revision 1

Both analyses moved. The disagreements are now down to **zero**, and two of
them resolved by each side conceding to the other.

| Item | Rev 1 position | Now |
|---|---|---|
| Numbering | Two conflicting schemes (P1–P8 vs F16–F21) | **Settled on Claude Code's F16–F22.** Its renumbering is correct and I adopt it wholesale. |
| One-ended claims | CC: extend the quad (`threat: null`). Cowork: node annotation. | **CC conceded.** Node annotation adopted — frozen edge comparators stay untouched. |
| Internal alignment | CC: re-tune the weight. Cowork: raise the severity ceiling. | **CC conceded.** Severity ladder adopted; weight untouched until F16 lands. |
| Unattached hazard | CC: teach the `worsens` self-loop. Cowork rev 1: **sided with CC** and deferred my own node flag. | **I now reverse myself.** My reason for deferring was wrong — see §3. Node flag it is. |

**The numbering collision CC caught, and I did not.** `P1`–`P6` are already
taken: they are the Loop 1 perception checks in `repair_loop.py`. My rev 1
happily wrote "the check list becomes P1–P7" while *also* carrying a separate
P-series from CC's draft. CC spotted the fork and fixed it. Its mapping table
is the authoritative one.

**Mapping from my rev 1 numbers, so nothing is lost:**

| Cowork rev 1 | Final |
|---|---|
| F16 dot | **F16** |
| F17 internal alignment stricter than Arm A | **F20** (merged with CC's) |
| F18 alignment probing | **F19** |
| F19 self-loops (victim) | **F18** (merged with the hazard direction) |
| F20 unattached hazard | **F18** (same finding, other direction) |
| F21 null path + prompt-forced fabrication | **F17** — the slot CC deliberately left open |
| F22 caption | **F21** |
| F23 severity ladder | folded into **F20** |
| — | **F22** — Graph B is right where Graph A is wrong (CC's promotion) |

---

# 2. The convergence log

Worth recording, because it is the point of running two analyses.

**Claude Code found, and I missed:** Graph B as the in-run control; Arm A's
OR vs our AND (`main.py:1700`); the correct layer for the caption check
(Stage 1, not Stage 2); the numbering collision; the batching economics; the
missing rung in my disposition ladder (§3); the elastic-yardstick question
inside F19 (§4); the true cost of probing B (30 calls per pass).

**I found, and Claude Code missed:** the null path — F_park_control scores
0.448, *below* the burning house, and CC's rev 1 never mentioned the control
scene at all. CC now records that as a miss rather than back-filling it, which
is the right call. Also: that the safe-scene fabrication may be **prompt-forced**
rather than a model defect; the node-annotation representation; the severity
ladder; that Graph A ×5 is free; the 371-test audit.

**Both of us, independently, on the same evidence:** the dot, the once-only
alignment, the self-loop, the caption miss. Four for four. That convergence is
the strongest reason to trust the diagnosis.

**CC also wrote its own reconciliation** (`..._reconciled_claudecode.md`),
deliberately without reading mine — two independent reconciliations beat one
contaminated by the other, which is the same reason the two source analyses were
worth having. It re-derived my factual claims from the run records rather than
accepting them, and confirmed every cell of the six-scene table. It found one
error and one design flaw in my proposal; both are now folded in above (the test
count, §2 below; the confounded F17b experiment, F17).

**The test count, settled precisely — we were both right.** I wrote 371, CC
wrote 383. `grep` for test functions gives **371 definitions**; three of them
are `@pytest.mark.parametrize`d (7 + 4 + 4 cases), so pytest collects
371 − 3 + 15 = **383**. I will say "371 test functions, 383 collected cases" from
here on, because the change-budget arithmetic in F16 and F18 counts *functions*.

---

# 3. The last conflict, resolved with code rather than argument

My rev 1 deferred the node-flag treatment of an unattached hazard on this
reasoning: *"a hazard with zero edges may not be reachable by frozen
`pick_suppression_framework`, so stripping its only edge could silently remove
it as a suppression target."*

**I checked. That is wrong, and it reverses the conclusion.**

`main.py:1290` builds `intervention_candidates` from `hazard_bearing` **nodes**,
not from edges:

```python
intervention_candidates = [
    {"threat": tid, "state": st,
     "outgoing_edge_count": sum(1 for e in edges
                                if e["source"] == tid and e["via_state"] == st)}
    for tid, st in sorted(hazard_bearing)
]
```

A hazard with zero edges is still a candidate, with `outgoing_edge_count: 0`.
It is never dropped. So removing the `worsens` self-loop costs it **rank, not
existence** — and `pick_suppression_framework` sorts by outgoing edge count
descending, so an unattached hazard will sort last and will usually fall out of
`top_n=3`. That is a real consequence to name in the UI, but it is not
erasure. **The node-flag version wins.**

**And the check turned up something better.** Frozen Arm A *already* models
this exact case, and treats it as an observation rather than a violation:

```python
orphan_threats = [... for c in intervention_candidates if c["outgoing_edge_count"] == 0]
threat_reasoning_coverage = reasoned_threats / total_threats
```

with the comment: *"A declared threat with zero outgoing edges is a Layer 2 →
Layer 3 asymmetry: the model recognizes the hazard but does not reason about
its causal reach."*

So **Arm B's `hazardous_node_no_edges` violation is stricter than frozen Arm A,
which records the same fact as a coverage metric.** That is the *second* place
we have found Arm B more punitive than the arm we are supposed to be
comparable to — the first being the AND/OR leg CC found at `main.py:1700`.
Two instances is a pattern, and it belongs in the ledger as a note on F20:
**where Arm B and Arm A disagree about strictness, Arm A is the reference, and
the burden is on us to justify any extra strictness.**

---

# 4. The findings, final form

## F16 · The `·` separator carries two meanings

**Problem.** `recommend.py:272` emits `at_risk: hazmat_worker_1·proximity`
(id·**role**) while five other sites emit id·**state**. The model generalised
from the syntax: D_aerial's `affected_objects` are `["hazmat_worker_1·proximity",
…]`; B_pool's quad carries `"state": "proximity"`. Damage: Graph A targets stop
being node ids → `unresolved_endpoint` at severity 2, **category fabrication** —
we charge the model for our own string; `_topological_edge_key` resolves the
dotted id off-vocabulary → A-vs-B collapses to 0.00 under the **top trust
weight, 0.30**. D_aerial's 0.0 alignment is an artefact, not a finding.

**Solution.** (1) Prompt — role gets a named key:
`child_2: child, state=swimming (normal), at_risk_as=proximity`. (2) Prompt —
quad spec: *`state` must be one of the `state=` values listed for that
object_id; `distress` and `proximity` are at-risk ROLES, not states.*
(3) Code — collapse the three strip helpers (`recommend.py` 346, 381, 520) into
one `bare_id()` and apply it at **every** comparison boundary: into
`build_causal_graph`, into `internal_alignment`, into `ab_alignment` — a
sanitising layer **in front of** frozen Arm A, never inside it. (4) UI —
`child_2 [swimming] · at risk: proximity`.

**What else changes.** Prompt: 2 edits. **Rulebook text: one line** on the quad
format (CC's catch — code stops firing, rulebook has to teach why). Conformance
code: none. Alignment formula: none. Frozen Arm A: untouched. **Existing tests:
none** — verified across all 371 functions; no test asserts `_scene_block` content or
`at_risk: id·kind`, and `test_recommend.py:215` already covers dot tolerance.
New tests: `bare_id` table; graph-A target resolution; no double-fire; A↔B match
through a dirty rec; and a guard that a genuinely hallucinated `x_9` **still**
raises `unresolved_endpoint`. Flowchart: no change.

## F17 · No null path, and the fabrication may be prompt-forced

*Not found by Claude Code; the slot is reserved for it in its rev 2.*

**Problem (a) — arithmetic.** F_park_control is safe: no disaster, no hazard
node, Graph B empty. `ab_alignment` therefore takes penalty **1.0** — nothing
to agree with reads as total disagreement — at weight 0.30; `pick_agreement` is
0.333 because a_pick is empty, b_pick says `"None"`, llm_pick is not. Trust
**0.448**, below the burning house, and all six scenes land in one 0.15-wide
band. **The silence test failed, and it failed on our arithmetic.**

**Problem (b) — the fabricated `dog_1 exposes swing_1` may be ours.**
`RECOMMEND_PROMPT` explicitly permits `assumptions_advisory` to be empty and
**never** permits `recommendations` to be empty, while framing the task as
"produce emergency-response recommendations for THIS scene."

**CC's F22 table sharpens this considerably.** On the same safe scene, Graph B
returned `[]` with *"no hazards present to suppress"* — **correct and silent** —
while Graph A fabricated. The model can say nothing when asked directly. It
only fabricates on the path where our prompt does not permit silence. That is
now strong circumstantial evidence for (b).

**Solution (a).** When there is no disaster and no hazardous node, signals with
nothing to compare are marked **not applicable** and dropped from the weighted
average, weights renormalised — not scored zero. Recorded with a note and an
event, so nothing is silent. Three empty picks is **agreement**, not
disagreement.

**Solution (b) — permission, not steering.** Say: `recommendations: array —
MAY BE EMPTY. If nothing in the scene is hazardous, return [].` Do **not** say
"do not invent hazards" or "be conservative" — that teaches the answer, breaks
iron rule 5, and Goodharts the measurement: we would fix the behaviour and
destroy our ability to observe it in one move. **Give it a legal way to be
silent, never a reason to be.**

**What else changes.** Prompt: one clause. Rules: every conformance and
internal-alignment check must tolerate an empty recommendation set without
raising. Code: the trust roll-up gains an `applicable` flag per signal and
renormalises; `pick_agreement` gains a null case. Tests: a safe-scene fixture
asserting trust lands **high**, N/A signals listed as such, and an empty rec
list raising no issue. **Flowchart: yes** — a null branch on the TRUST box.

**Correction from CC's reconciliation — my experiment for (b) was confounded,
and I accept the fix.** I proposed adding the clause and re-running F_park to
see whether the fabrication disappears. But the clause sits in the **shared**
prompt, so it changes all six scenes at once. A model newly permitted to say
nothing may also say *less* on the hazard scenes — the very change that makes
F readable could move the baseline I am reading it against.

**Run F_park paired instead:** the same scene twice in one session, with and
without the clause, everything else fixed. That isolates the prompt effect from
the scene effect with one extra run. The clause only enters the shared prompt
after the pair reads clean. **Consequence for the ledger:** part (b) is not a
finding yet. CC files it as **O1, an open question**, and that is the honest
status — I file it the same way. F17 as a *finding* is part (a), the
arithmetic, alone.

## F18 · One-ended causal claims have no representation

*Merged: victim-with-no-source and hazard-with-no-target are the same defect
from two directions. CC's merge, and it is right.*

**Problem, direction A — victim, no source.** The schema forces every at-risk
entity to be the `affected_object` of a quad, and every quad needs a threat.
"Rescue the trapped person" has no legal form, so the model makes the victim
its own threat: `child_1→child_1`, `person_1→person_1 via 'trapped'` while
`building_1 · collapsed` sits unused.

**The proof is CC's, and it is the strongest single piece of evidence in the
round.** Same model, same scene, same run:

| Graph A (from rec quads) | Graph B (asked directly) |
|---|---|
| `child_1 → child_1`, `child_2 → child_2` | `pool_1 → child_1`, `pool_1 → child_2` |

**Graph B is the in-run control.** F3 proven rather than hypothesised.

**Problem, direction B — hazard, no target.** `spill_1` in C and D:
`hazardous_node_no_edges … (needs a target or a worsens self-loop)`. The rule
names a legal form the prompt never teaches. And per §3, **frozen Arm A records
this same fact as `orphan_threats` + `threat_reasoning_coverage`, an
observation, not a violation.** We are stricter than the reference arm.

**Solution.** A one-ended claim is a **node annotation**, not an edge:

```
node spill_1  { hazardous: true, unattached: true,   note: "no target named" }
node person_1 { at_risk: true,   unattributed: true, note: "no source named" }
```

The model emits it as `threat: null` / no `affected_objects` with an explicit
marker (CC's emit form); the code turns that into a node flag rather than an
edge (my representation). A null-threat quad **never becomes an edge**, so
`build_causal_graph`, `compare_graphs_topological` and
`pick_suppression_framework` never see a shape they do not know. Scored as a
**gap (severity 0–1)**, never a fabrication. Becomes a reflection trigger:
*"you named a victim with no cause — what caused it?"* We ask; we never name
`building_1`. Prompt neutrality holds.

**Disposition ladder — with CC's missing rung added.** This closes the loop
instead of nagging forever:

| State | Severity | Pressure |
|---|---|---|
| gap, never asked | 1 | trigger reflection once |
| gap, asked, model answered "none visible", **and no `hazard_bearing` entity exists in the record** | **0** | closed — **recorded**, not erased (rule 8) |
| gap, asked, model answered "none visible", **but a `hazard_bearing` entity does exist** | **inconsistency, not a terminal answer** | stays open |
| gap, asked, model named a source | recorded as revision | closed |

**Row 3 is CC's catch and it is important.** On B_pool the code *derives*
`pool_1` as hazard-bearing via the medium-bound rule. A model that then says
"no source visible" for a drowning child is contradicting the shared record,
not making an honest declaration. Without that rung, "none visible" becomes a
dodge that scores 0 — we would have built an escape hatch and called it
honesty.

**Ranking consequence to disclose (§3).** An unattached hazard has
`outgoing_edge_count: 0`, so `pick_suppression_framework` sorts it last and it
will usually fall outside `top_n=3`. It is still a candidate, never dropped.
The UI must show it as *declared hazardous, no target named, not in the top
suppression set* rather than letting it vanish.

**What else changes.** This is the expensive one. Rules: `edge_from_non_hazardous`
and `via_state_not_hazard_bearing` stop firing on victim self-loops because
those edges no longer exist (B_pool loses 4 of its 9 issues); add
`unattributed_victim`; `hazardous_node_no_edges` becomes `unattached_hazard`
at Arm A's severity, i.e. observational. **The hazard/victim disambiguation
must be explicit in both prompt and rules** (CC's warning): a `worsens`
self-loop on a **hazard-bearing** node is legal and meaningful; a `may_harm`
self-loop on an **at-risk** node is not — same shape, opposite meaning,
disambiguated by node kind. Prompt: relax `affected_objects: NON-EMPTY` to
permit the declared-unattributed form. Reflection: new trigger family with the
ladder above. **Existing tests:
`test_internal_alignment_state_mismatch_and_self_loop` and the self-loop
conformance tests must be rewritten** — the only finding where existing tests
move. Alignment numbers **will** move on re-run; that is intended, and it is a
second reason not to fix the weights yet. Flowchart: new step in the graph-A
build box, new trigger family on the Stage 4 reflection box.
**Needs explicit ratification.**

## F19 · A-vs-B is a point estimate

**Problem.** The 5 probes measure **recommendations only**. Graph B is asked
**once** at temp 0; Graph A is built **once** from the temp-0 recommendations,
even though the probes produced 5 distinct recommendation sets in all six
scenes. A one-shot number on a base that wobbles every re-ask, reported to
three decimals, under the top trust weight. Same defect class as F15.

**Solution.** Probe both sides.

- **Graph A ×5 — free.** The 5 probe recommendation sets already exist and are
  discarded after reduction to a flat reading. Build a Graph A from each. Zero
  model calls.
- **Graph B ×5 — 5 calls per scene, 30 per calibration pass** (CC's costing).
- Report **median + spread with drivers** (CC's spec) **and per-edge belief
  rate** (my diagnostic):

```
A-vs-B  structural 0.40  (spread 0.20–0.55 over 5×5)
asserted, not believed
  house_1 may_spread_to house_2   believed in 0/5 B-graphs   ← faithfulness failure
  car_1 blocks_access_to house_1  believed in 3/5 B-graphs   ← noise
```

**The second question hiding inside this — CC's, and it deserves its own
line in the ledger.** Probing Graph B measures something never yet measured:
**is the model's causal belief itself stable?** Today B is the yardstick for A
and is assumed fixed. If B wobbles 5/5, the yardstick is elastic and every
A-vs-B number in this project inherits that. This is not plumbing; it is a
research question about whether the control is a control.

**Sequencing consequence.** Because the A side is free and the B side is 30
calls, **split the finding**: A×5 goes into batch 1 at zero cost and already
converts the point estimate into a distribution against a fixed B; B×5 follows
the first re-run. See §5.

**What else changes.** Prompts: none. Rules: none. Code: retain the probe rec
sets; loop `build_graph_a`, then `run_graph_b`; an `ab_alignment_distribution`
wrapper around the **unchanged** single-pair `ab_alignment`. Trust folds the
median, not the point; the penalty becomes belief-rate weighted rather than
count-based. Tests: existing `ab_alignment` tests stay valid; new tests for the
wrapper with a deterministic fake probe function — hermetic. Flowchart: GRAPH B
box gains "×5 probes"; A-vs-B box becomes "alignment distribution".

## F20 · Internal alignment counts trivia, and is severity-blind

**Problem, two causes of one symptom.** (a) Arm B enforces `reason id ∈ related
AND ∈ quad`; **Arm A relaxes the same leg to OR** (`main.py:1700`). The model
leaves `related_objects` empty, so it fires on nearly every recommendation — we
penalise an unfilled optional field. (b) The dot makes `quad_ids` and
`reason_ids` disagree, so both the coverage branch and the strict branch fire on
one cause. F15's double-counting in a new layer.

**And once those clear, what remains is genuinely mixed:**

| Failure | Trivial? |
|---|---|
| `quad state 'proximity' != child_2's state 'swimming'` | No — contradicts the frozen record. |
| `at-risk child_1 used as a threat` | No — **role inversion**, the exact causal-direction error CEE+ exists to catch. |
| `rec 3: remaining_risk duplicates rec 2` | Yes — severity 0 is right. |

**Solution.** Match Arm A's OR (or fire only when `related_objects` is
non-empty); F16's `bare_id` removes cause (b). Then **keep the 0.20 bucket
weight and raise the severity ceiling: role inversion becomes severity 3**,
above the current maximum. The weight answers "how much does internal coherence
matter"; the severity answers "how bad is *this* incoherence."

**Weight note (both of us agree).** The 0.20 reads low today, which is
*accidentally* correct because roughly half its failures are artefacts. Do not
re-tune until F16 lands and the panel is clean.

**Ledger note from §3.** Two independent cases now of Arm B being stricter than
frozen Arm A — the AND/OR leg here, and `hazardous_node_no_edges` vs Arm A's
`orphan_threats`. Record the principle: **Arm A is the strictness reference;
extra Arm B strictness must be justified, not inherited by accident.**

**What else changes.** Rules: predicate + severity constants. Prompts: none.
Tests: existing assert categories, so they hold; add one for empty
`related_objects` and one asserting role inversion outranks a duplicate.
Flowchart: no change.

## F21 · Caption contradiction is invisible to a self-consistent pipeline

**Problem.** B_pool caption: *"another child floats motionless and unconscious
face down farther away."* Record: `child_2 · swimming (normal)`. `swimming` is a
legal state so the vocab check did not fire; `child_2` was present so the
missing-entity check did not fire. The apparatus then reported **U = 0.033** —
the lowest of any hazard scene — `self_confidence 0.95`, zero violations,
**zero reflection rounds, reason "clean."**

Reflection and petition were **right not to fire**. **Every pressure signal we
own is a self-consistency signal, and a model that is consistently wrong passes
all of them.** F5 in its purest observed form.

**CC's three-layer trace — the information was never lost by the model, only by
the record:** Stage 1 `child_2 · swimming (normal)`; Stage 2 reason *"The child
is swimming near an engulfing hazard, which **could** lead to a similar
situation **if not supervised properly**"*; Stage 4 rank 2 *"**Rescue the
unconscious child** from the water."*

**CC's layer attribution, adopted, and it matters for the ledger.** Stage 1 is a
plain perception error (**Category E**) — a bare wrong word, no reasoning,
nothing rationalised. The **Stage 2 reason string** is where rationalisation
lives; those two conditional hedges converting an unconscious victim into a
supervision concern are a textbook minimisation signature. **Do not label the
perception layer a pathology.** Over-labelling now is what would make S5
meaningless later.

**Solution.** Placement is CC's — **Loop 1, Stage 1**, upstream of the bad
record, as check **P7**. Safety rails are mine.

- **(a) Detector.** The caption carries a condition word for an entity whose
  declared state is a non-condition state. Quote the **caption's own words** and
  ask the model to reconcile. It may STAND — that is evidence either way.
- **(b) Observation.** Record the **direction**: caption severity > declared
  severity = a **downgrade**. A minimisation signature, and the first one
  observable at the perception layer. Becomes a row in the pathology trace when
  S5 lands.
- **Rails.** It may flag **only terms absent from the record**, and may **never
  propose which state to use**. The lexicon lives in `vocabulary.py` and draws
  on the existing closed state vocabulary.
- **Prompt neutrality.** Legal under rule 5: we quote the **given input**, never
  our reading of the scene — the same grammar `caption_entity_missing` uses.

**What else changes.** Vocabulary: a caption condition-word lexicon. Rules: new
rulebook rule (statement + rationale + example). Petition: new kind in
`_PETITIONABLE`, routed to **stage-1**. Tests: present-in-record → no trigger;
absent → trigger; empty caption → no trigger; **off-vocabulary word → no
trigger** (guard against the extractor inventing states). **Flowchart: yes** —
Loop 1's check list becomes **P1–P7**.

## F22 · Graph B is right where Graph A is wrong

*CC's promotion of its own closing note to a finding. Correct — it reframes
what A-vs-B measures.*

| Scene | Graph B (asked directly) | Graph A (from rec quads) |
|---|---|---|
| B_pool | `pool_1→child_1`, `pool_1→child_2` — **correct** | `child_1→child_1`, `child_2→child_2` |
| F_park | `[]` + *"no hazards present to suppress"* — **correct** | fabricates `dog_1 exposes swing_1` |
| D_aerial | `tanker→spill→hazmat_worker` chain — **correct** | 8 flat edges, all `·proximity` |

Same model, same scene, same run. **Asked "what causes what," the model is
reliably right. Asked to justify a recommendation as a quad, it degrades.**

**Why it outranks the individual defects.** CEE+'s thesis is that
recommendations are declaratively coherent but not causally grounded. This
batch suggests something sharper: **the model holds the causal structure; the
recommendation schema is where it is lost.** So A-vs-B disagreement is
currently a blend of genuine ungroundedness and our own schema corrupting the A
side, and F16 + F18 must land before the two can be separated.

**Consequence for publication.** No Stage-4 grounding claim is safe until that
separation is done. Graph B should be documented as the **in-run control** —
same model, same scene, same run, with the recommendation schema removed as the
only variable.

**One caveat I would add to CC's version.** F22 rests on three scenes, and on
Graph A being corrupted by F16 in at least two of them. It is possible that
after F16 and F18 land, Graph A recovers and F22 dissolves. **That is the whole
point of the re-run**, and it is pre-registered in §6. Until then F22 is a
strong hypothesis, not a result — it should not be written into the ledger as
a settled finding.

---

# 5. Punch list (not ledger entries)

- **Relabel the intervention panel:** "suppression target (for the causal
  test)", with the operational recommendation shown separately. Suppressing
  `pool_1` is a correct *probe* and a terrible *instruction*. Load-bearing once
  S6 lands.
- **Near-duplicate recommendations:** D_aerial's two recs share threat and
  affected set, differing only in the effect verb. Already detected; surface as
  an uncertainty driver, carry into Stage 4 reflection in Phase 2. Low priority.
- **Both D picks display `police_car_1`:** display artifact — the panel draws
  only the first affected object. List all of them.
- **`spill_1` clicks through to uncertainty only:** zero edges, in no
  recommendation. F18's node flag gives it a real card.
- **"a threat in only 0/5 re-asks":** 0 is literal and should not read the same
  as 3/5. Proposed: a *singleton claim* flag and a per-rec trust floor at 0/5.
- **"Bound 4/6":** grounding quality, not a defect count — 4 entities got a
  DINO box, 2 fell back to VLM box + SAM.

---

# 6. Final sequencing

Re-runs are the expensive part (live, needs Ollama), so batch everything
mechanical into one pass. This is CC's economics, with F17 inserted and F19
split.

**Batch 1 — cheap, additive, no existing test rewrites**

F16 (dot, all four parts) · F17**a** (null path arithmetic only) · F19 **A
side only** (free — 5 Graph As from the existing probe sets) · F20 (OR not AND,
severity ladder) · F21(a) (caption detector) · F18's **prompt line** for the
orphan-hazard direction · the panel relabel.

**The empty-rec clause is no longer in batch 1.** It is the one change that
touches the shared prompt, so it goes in the paired run below instead — my rev 1
had it riding along with everything else, which would have moved the baseline
underneath the other five scenes.

**Then: one re-run of all six, plus one extra paired F_park.** Every alignment
number currently held is suspect. The three clean experiments land on different
scenes and do not confound each other: **B** answers whether the caption check
fires; **D** answers whether the dot was the whole cause of the alignment
collapse; **F** answers whether the null-path arithmetic alone lifts the safe
scene above the hazard scenes. **O1 — whether the fabrication is prompt-forced —
is answered by the paired F_park pair only** (same session, clause on vs clause
off, everything else fixed), and the clause enters the shared prompt only if
that pair reads clean.

**Then F19's B side** — 30 calls per pass, and the first measurement of whether
the yardstick itself is stable.

**Then F18** — the expensive one; rules and tests both move. **Ratification
first.**

**Only then set the trust weights.** F16, F17, F18, F19 and F21 all change what
the numbers mean. Tuning now would be tuning against a corrupted signal.

---

# 7. Pre-registered predictions

Recording these before the re-run turns it from a look into a test. CC's list,
plus the F17 predictions it did not have.

| # | Prediction | Falsifies |
|---|---|---|
| 1 | D_aerial alignment **moves off 0.0** | if not, F16 was not the cause |
| 2 | B_pool alignment **stays low** — the self-loops survive until F18 | if it jumps, the self-loops were not the driver |
| 3 | B_pool's caption check **fires** on "unconscious"/"motionless" | if not, the detector's rails are too tight |
| 4 | A_fire and D_aerial internal alignment **rise** (F16 + F20) | if not, the trivia was not the bulk |
| 5 | **F_park trust rises above every hazard scene** (F17a) | if not, the null path is not the only arithmetic defect |
| 6 | **F_park returns `recommendations: []` in the clause-on half of the pair, and still fabricates in the clause-off half** (O1) | if it fabricates in both, it is genuinely the model's and is real pathology material; if it returns `[]` in both, the clause was never the cause |
| 6b | The clause-on run of the **five hazard scenes** produces the same recommendation count as clause-off | if counts drop, the clause suppresses output generally and must not ship |
| 7 | Graph B ×5 is **stable** (F19, second pass) | if it wobbles, the control is elastic and F22 weakens |

Prediction 6 is the one I most want to be wrong about, because a model that
fabricates urgency **with an escape hatch open** is a far more interesting
result than a prompt bug. Prediction 6b is the guard CC's paired design bought
us: without it, a quieter model looks like a fixed one.

---

# 8. Flowchart edits (iron rule 9)

1. **Loop 1 box** — check list becomes **P1–P7**; add "caption-state
   contradiction (caption's condition word vs declared state)."
2. **TRUST box** — add a null branch: "no disaster and no hazard node → signals
   with no comparand marked N/A and dropped; weights renormalise."
3. **GRAPH B box** — add "×5 probes."
4. **A-vs-B box** — becomes "alignment distribution (median + spread + per-edge
   belief rate)."
5. **Graph-A build box** *(F18 only, after ratification)* — add "attach, or
   declare unattached / unattributed."
6. **Stage 4 reflection box** *(F18 only)* — add the one-ended-claim trigger
   family.

Edits 1–4 go with batch 1; 5–6 wait for F18.

---

# 9. What needs Sunny's explicit yes before any code

1. **F18** — it extends the ontology, even though the extension stays Arm B-side
   and frozen comparators never see it. Iron rule 3.
2. **The extra rung** in F18's disposition ladder (§3): a "none visible" answer
   becomes an *inconsistency* rather than a terminal answer when code has already
   derived a hazard into the record.
3. **O1's prompt clause** permitting an empty recommendations array — and,
   separately, whether it ships to the shared prompt after the paired F_park run
   reads clean. It is the one change that could alter model behaviour on every
   scene, and it is deliberately an experiment rather than a fix.

Everything else in batch 1 is mechanical: it changes our own strings, our own
arithmetic, and our own strictness relative to frozen Arm A.

**Seven findings and one open question. Five are ours (F16, F17a, F18, F19,
F20), one is the model's (F21 — we showed it "motionless and unconscious face
down" and it returned `swimming`), one reframes the rest (F22), and O1 waits on
the paired run.** CC is right that the exact count is stronger than the round
one: "five of seven, plus a real model deficit" says more than "six are ours,"
because F21 is the finding that shows the witness has genuine deficits and that
every pressure signal we own is blind to a model that is confidently and
reproducibly wrong.

**Fixing the interview, not the witness — third stage running, with one
documented exception.**
