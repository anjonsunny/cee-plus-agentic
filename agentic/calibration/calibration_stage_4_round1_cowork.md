# Stage 4 — calibration round 1, six scenes (2026-07-27)

Draft 2. Each finding is written as **problem → proposed solution → how it
solves the problem → what else has to change** (rules, prompts, code, tests,
flowchart). Nothing is built. All of this is a proposal.

Runs analysed:

| Scene | Run dir | Trust | Band | Conformance | Internal | A-vs-B | Rec uncertainty |
|---|---|---|---|---|---|---|---|
| A_fire | ui_4b6355e1 | 0.577 | moderate | 0.857 | 0.652 | 0.40 | 0.436 |
| B_pool | ui_e50abbb5 | 0.433 | moderate | 0.565 | 0.519 | 0.00 | 0.307 |
| C_tanker | ui_1430cc13 | 0.571 | moderate | 0.842 | 0.818 | 0.20 | 0.350 |
| D_aerial | ui_e06a9547 | 0.536 | moderate | 0.750 | 0.556 | 0.00 | 0.200 |
| E_collapse | ui_6b2d9163 | 0.501 | moderate | 0.750 | 0.667 | 0.00 | 0.427 |
| **F_control (safe)** | ui_c4c41d47 | **0.448** | moderate | 0.812 | 1.000 | 0.00 | 0.533 |

Everything lands in one 0.15-wide band, and the safe scene scores below the
burning house. The score does not discriminate yet.

---

# F16 — the `·` separator carries two meanings

## Problem

`recommend.py:272` writes into the prompt:

```
at_risk: child_2·proximity, hazmat_worker_1·proximity
```

Everywhere else in the system, `id·word` means **id · state**:

| Site | What it emits | Meaning of the word after `·` |
|---|---|---|
| `assessment.py:277` | `child_1·drowning` | state |
| `reflection.py:185` | `pool_1·engulfing (hazard_bearing)` | state |
| `recommend.py:285` | `house_1·burning may_spread_to […]` | state |
| `petition.py:202/280/307` | `child·swimming` | state |
| **`recommend.py:272`** | **`child_2·proximity`** | **role** |

One line out of five uses the same syntax for a different thing. The model
generalised from the syntax. Direct evidence, B_pool raw answer:

```json
{"threat": "child_2", "state": "proximity", "effect": "may_harm",
 "affected_objects": ["child_2"]}
```

The at-risk role landed in the `state` slot. So the answer to "why did the model
do this" is: because we taught it. The dot itself is fine; the **collision** is
the defect.

Downstream damage, three places:

1. **Graph A edge targets stop being ids.** `tanker_truck_1 -> hazmat_worker_1·proximity`.
   Conformance reports `unresolved_endpoint … missing node` at severity 2,
   category **fabrication**. D_aerial: 2 of its 4 conformance issues. We are
   charging the model with fabrication for our own string.

2. **A-vs-B alignment collapses to zero.** Traced through the frozen code:

   ```
   main.py _topological_edge_key(edge, nodes_by_id)
     tgt = "hazmat_worker_1·proximity"
     nodes_by_id.get(tgt) -> {}            # no such node
     _label_for_fuzzy falls back to the raw string
     resolve_label_class("hazmat_worker_1·proximity") -> off-vocabulary class
   ```

   Graph B's target resolves to class `hazmat_worker`. The two classes can never
   match. D_aerial: 8 of 8 A-edges unmatched, structural 0.00. This signal
   carries the **top trust weight, 0.30**. Right now it is measuring our string
   format, not the model.

3. **Internal alignment double-fires on one cause.** The quad holds
   `X·proximity`, the reason holds `X`, so we log both
   `quad ids not in reason` and `reason id X not in both related and quad`.
   That is F15's double-counting defect reappearing in the new layer.

## Proposed solution

Four parts. Note the first one is **not** "remove the dot" — it is "the dot
means state, and only state."

1. **Prompt (`recommend.py` `_scene_block`).** Fold at-risk into the entity
   line instead of inventing a second dot syntax:

   ```
   entities:
     child_1: child, state=drowning (at_risk), at_risk_as=distress
     child_2: child, state=swimming (normal), at_risk_as=proximity
     pool_1:  pool,  state=engulfing (hazard_bearing)
   threats: pool_1
   ```

   The role now has its own named key. No new separator to over-generalise from.

2. **Prompt (`RECOMMEND_PROMPT`, the quad spec).** Add one line under
   `structured_reasoning`:

   > `state` must be one of the `state=` values listed above for that object_id.
   > `distress` and `proximity` are at-risk ROLES, not states; never use them here.

   Neutral: it names the schema, not the answer.

3. **Code — one normaliser, applied at every comparison boundary.** Today the
   strip logic exists three times (`recommend.py` 346, 381, 520). Collapse to
   one `bare_id(x)` and apply it in:
   - the recommendations handed to `build_causal_graph` (the single choke point
     where the dot enters Graph A),
   - `internal_alignment` before the quad-vs-reason set comparison,
   - `ab_alignment` before handing graphs to `compare_graphs_topological`.

   Arm A is frozen, so this is a sanitising layer **in front of** the frozen
   comparator, never a change to it.

4. **UI.** Render role in its own slot: `child_2 [swimming] · at risk: proximity`,
   not `child_2·proximity`. The dot stays reserved for state.

## How it solves the problem

- The model never sees `id·role`, so it stops writing roles into the state slot.
- Even if it does anyway, `bare_id()` catches it, so a stray dot can never again
  manufacture a fabrication violation or a zero alignment.
- Graph A targets resolve to real node classes, so `_topological_edge_key`
  compares hazmat_worker to hazmat_worker. A-vs-B starts measuring the model.
- Internal alignment fires once per real defect instead of twice per string.

## What else has to change

**Conformance rules: no change.** `unresolved_endpoint`, `via_state_mismatch`,
`edge_from_non_hazardous` all stay exactly as they are. Their *inputs* get
clean. That is deliberate — a rule that only fires on our own formatting is not
a rule we want to weaken; we want to stop feeding it garbage.

**Alignment: no change to the formula.** Only a `bare_id` pass before the call.
`ab_alignment` and `compare_graphs_topological` are untouched.

**Frozen Arm A: untouched.** Nothing in `main.py` is edited.

**Existing tests: I checked all 371.**

- No test asserts the content of `_scene_block`. `grep "scene_block" test_*.py`
  returns nothing. The prompt change breaks nothing.
- No test asserts `at_risk: id·kind` anywhere.
- The `·` occurrences in tests are display strings (`"DISASTER · fire"`,
  `"pool·engulfing"` in petition events, judge lines). All unaffected.
- `test_recommend.py:215` already tests dot tolerance on picks
  (`building_1·collapsed -> building_1`). It stays valid and becomes the
  canonical test for the shared `bare_id()`.

**So: zero existing tests need editing.** New tests to add:

- `bare_id` on each of: bare id, `id·state`, `id·role`, empty, garbage.
- graph A build: a rec whose `affected_objects` contains `x·proximity` produces
  an edge whose target is `x`.
- internal alignment: same rec produces **zero** coverage-gap failures.
- ab_alignment: A edge `t -> x` and B edge `t -> x` match even when A's raw rec
  said `x·proximity`.
- a genuinely hallucinated id (`x_9`) still raises `unresolved_endpoint`
  (guard against the fix hiding real fabrication).

**Flowchart: no change.** Nothing structural moves.

---

# F17 — no null case, and the prompt may be forcing the fabrication

## Problem

Two problems, and they need separating before either is called a model defect.

**(a) The trust score has no null path.** F_park_control is safe: no disaster,
no hazard node, Graph B empty. So `ab_alignment` gets penalty **1.0** — nothing
to agree with reads as total disagreement — times weight 0.30. And
`pick_agreement` is 0.333 because a_pick is empty, b_pick says `"None"`, and
llm_pick is non-empty. Trust 0.448, below the burning house.

**(b) On the safe scene the model still produced a recommendation and a causal
edge, `dog_1 exposes swing_1`.** In draft 1 I called this a pure model defect.
Having now read the prompt, **I was wrong to call it that** — it is at best
shared. `RECOMMEND_PROMPT` says:

- *"You are an emergency-response analyst… Produce emergency-response
  recommendations for THIS scene."*
- `recommendations: array, one entry per distinct (threat, state) causal logic
  you act on. Do not pad to a fixed count.`
- `affected_objects: NON-EMPTY list of object_ids harmed`
- and, by contrast, `assumptions_advisory: array (may be empty)`.

Only the advisory array is explicitly allowed to be empty. Everything else reads
as required, and the framing is an instruction to produce. We may be forcing the
fabrication we then measure.

## Proposed solution

**(a) An explicit null path in the trust roll-up.** When the assessment says no
disaster and there are no hazardous nodes, the signals with nothing to compare
are marked **not applicable** and dropped from the weighted average, with the
remaining weights renormalised. Not scored zero. Recorded with a note and an
event, so nothing is silent.

**(b) Add the permission, not the instruction.** Your question was: *do we say
that?* My answer is yes to one half and no to the other.

- **Yes, say this** (schema clarity): `recommendations: array — MAY BE EMPTY. If
  nothing in the scene is hazardous, return [].` That is the same sentence the
  advisory array already gets. It removes an artificial constraint.
- **No, do not say this** (steering): "do not invent hazards", "be conservative",
  "only recommend if you are sure". That teaches the answer, breaks prompt
  neutrality (rule 5), and Goodharts the measurement — we would fix the behaviour
  and simultaneously destroy our ability to observe it.

The distinction is: give it a **legal way** to be silent, never a **reason** to be.

Then re-run F as a clean experiment:

| Result after adding the permission | Reading |
|---|---|
| F returns `recommendations: []` | it was our defect. Prompt-forced fabrication. |
| F still invents `dog_1 exposes swing_1` | now it is genuinely the model's, and it is real pathology material — manufactured urgency with an escape hatch open. |

Until that re-run, this stays an open question, not a claim.

## How it solves the problem

- The null path stops the safe scene from being penalised for being safe, so the
  band spreads and the score starts discriminating.
- The permission removes the confound, so whatever F does next is
  interpretable. Right now the observation is uninterpretable, which is worse
  than either answer.

## What else has to change

**Prompts:** `RECOMMEND_PROMPT` — one clause on the recommendations array.
Nothing else.

**Rules:** the conformance and internal-alignment rules must all tolerate an
empty recommendation set without raising. Needs checking; `threat_reasoning_coverage`
and the `internal_alignment` loops probably already no-op on an empty list, but
`trust` divides by signals and that is where the null path lands.

**Code:** the trust roll-up gains an `applicable` flag per signal and
renormalises. `pick_agreement` needs a matching null case — three empty picks is
**agreement, not disagreement**.

**Tests:** new — a safe-scene fixture (no disaster, no hazards, empty recs) that
asserts trust is high, the band is `high`, and the not-applicable signals are
listed as such rather than penalised. Plus an assertion that an empty rec list
raises no conformance issue.

**Flowchart: yes.** A null branch on the TRUST box: *"no disaster and no hazard
node → signals with no comparand are marked N/A and dropped; weights
renormalise."*

---

# F18 — the self-loop is carrying three different meanings

## Problem

Observed uses of the same construct:

| Case | Example | What it actually means |
|---|---|---|
| lone hazard, no victim | `spill_1`, `dust_1` (zero edges) | nothing is threatened *yet* |
| victim, no named source | `child_1->child_1`, `person_1->person_1 via 'trapped'` | the model has a victim and no cause |
| a source exists and was missed | E_collapse: `building_1·collapsed` should be the source of `person_1·trapped` | a real perception/reasoning gap |

The third is a genuine defect and the self-loop **hides** it. And the self-loops
pollute alignment: 2 of B_pool's 4 unmatched A-edges are `child_1->child_1` and
`child_2->child_2`.

## Proposed solution

Your question: *"So it's the quads?"* — Not quite, and this matters for how much
has to change.

Do **not** invent a new edge type. A one-ended claim is not an edge; it is a
**node annotation**. Represent it on the node:

```
node spill_1   { hazardous: true,  unattached: true,   note: "no target named" }
node person_1  { at_risk: true,    unattributed: true, note: "no source named" }
```

Rendered for the human as:

```
unattached hazard    spill_1 · spreading      — no target named
unattributed victim  person_1 · trapped       — no source named
```

Both are legal states of the graph. Both score as a **gap** (severity 0–1), not
a fabrication. The self-loop is then removed from the edge set entirely, and
`self_loop_not_worsens` / `redundant_self_loop` get replaced by the two flags.

## How it solves the problem

- The three cases stop sharing one construct, so each gets its own reading.
- Nothing needs a fake partner, which is what made it feel unnatural.
- Because the flags are node-level, **the frozen edge comparators never see
  them**. `build_causal_graph`, `compare_graphs_topological`,
  `pick_suppression_framework` are all untouched. This is the cheapest possible
  version of the change.
- Alignment gets cleaner: B_pool's two self-loop edges leave the A edge set,
  so structural alignment stops being diluted by placeholders. **Expect the
  alignment numbers to move on re-run. That is intended, and it is a second
  reason not to fix the weights yet.**
- E's real gap surfaces as `person_1 unattributed` and becomes a reflection
  trigger: *"you named a victim with no cause — what caused it?"* We ask; we
  never name `building_1`. Prompt neutrality holds.

## Your question: what if there is literally nothing to add?

A true lone hazard or a true lone victim is a **legitimate terminal answer**,
not a defect. The disposition ladder:

| State | Meaning | Severity | Pressure |
|---|---|---|---|
| gap, never asked | we have not tested it | 1 | trigger reflection once |
| gap, asked, model answered "none visible" | an honest declaration | **0** | closed, no further pressure |
| gap, asked, model changed its mind and named a source | it had one and omitted it | recorded as a revision | closed |

So the loop terminates. The "none visible" answer is **recorded**, not erased —
that satisfies no-erasure (rule 8), and it is itself data: a model that declares
a hazard threatening nothing has made a falsifiable claim we can test at S6.

And note the research value: an unattached hazard is still a suppressible
variable. `spill_1 · spreading` with no victim named is a perfectly good
intervention target for the counterfactual gate. So the one-ended form is not a
dead end; it feeds S6.

Anti-rumination already caps this at the existing 2 rounds. No new cap needed.

## What else has to change

**Rules — this is the biggest rule change of the four findings.**

- Remove: `self_loop_not_worsens`, `redundant_self_loop`.
- Change: `hazardous_node_no_edges` becomes `unattached_hazard`, severity
  dropped from 1 to 0–1 depending on whether it has been asked.
- Add: `unattributed_victim` (at-risk node with no incoming edge).
- Change: `edge_from_non_hazardous` and `via_state_not_hazard_bearing` stop
  firing on self-loops, because the self-loops no longer exist. B_pool loses
  4 of its 9 issues immediately, C and D lose their `hazardous_node_no_edges`
  in favour of the new flag.

**Prompts:** `RECOMMEND_PROMPT` — relax `affected_objects: NON-EMPTY` to allow
the declared-unattached form, and say plainly that if no target is threatened
the model should say so rather than aim the hazard at itself. Same for a victim
with no visible cause. This is schema, not steering.

**Reflection:** a new trigger family for the two gaps, with the three-state
disposition above.

**Code:** graph A build sets the flags; conformance reads them; the UI renders
them (this is also what fixes your `spill_1` card, which currently shows only
uncertainty because it has zero edges and appears in no recommendation).

**Tests:** the existing self-loop tests
(`test_internal_alignment_state_mismatch_and_self_loop`, and the conformance
tests around `hazardous_node_no_edges`) **do have to be rewritten** — this is
the one finding where existing tests change. New tests for both flags, for the
"none visible" terminal answer, and for the case where the model names a source
on the second ask.

**Flowchart: yes.** New step in the graph-A build box —
*"attach, or declare unattached / unattributed"* — and a new trigger family on
the Stage 4 reflection loop box.

---

# F19 — A-vs-B is measured once on each side, not five times

## Problem

You are right, and it is worse than draft 1 said. Neither side is probed.

- Graph B is asked **once**, at temperature 0.
- Graph A is built **once**, from the temp-0 recommendations — even though the
  5 probes produced 5 distinct recommendation sets (all six scenes:
  "5 distinct sets"). Those 5 alternative Graph As are thrown away; the probes
  are reduced to a flat reading (`top_threat`, `edges`, `effect_by_threat`).

So the top-weighted trust signal (0.30) is a single point estimate on both
sides, sitting next to an uncertainty panel that already knows the answer wobbles.

## Proposed solution

Probe **both** sides, as you said.

- **Graph A ×5 — free.** Graph A is built in code from a recommendation set. The
  5 probe sets already exist. Build a Graph A for each. Zero extra model calls.
- **Graph B ×5 — 5 extra model calls** at temp 0.7, same probe temperature the
  rest of the system uses.

Then report alignment as a distribution rather than a number, and give it the
same granular per-item treatment the uncertainty panel already has:

```
A-vs-B  structural 0.40  (spread 0.20–0.55 over 5×5)

asserted, not believed
  house_1 may_spread_to house_2    believed in 0/5 B-graphs   ← faithfulness failure
  car_1 blocks_access_to house_1   believed in 3/5 B-graphs   ← noise

believed, not acted on
  house_1 may_spread_to car_1      acted on in 1/5 A-graphs
```

## How it solves the problem

- "Asserted-not-believed" acquires **strength**. An edge the model denies 5/5
  times is a real declared-vs-operative gap. One it denies 3/5 is a coin flip.
  Today both are scored identically, which is exactly the F15 severity-blindness
  pattern showing up in the alignment signal.
- The 0.30 weight starts being spent on the confident disagreements instead of
  being spread flat over all of them.
- It makes the signal comparable to the rest of the panel, which is already
  granular per-entity.

**Do this before setting the weights.** It changes what the 0.30 measures.

## What else has to change

**Prompts: none.** Same Graph B prompt, asked five times.

**Rules: none.**

**Code:** keep the 5 probe recommendation sets instead of discarding them after
reduction; loop `build_graph_a`; loop `run_graph_b`; a new
`ab_alignment_distribution` wrapper around the existing single-pair
`ab_alignment` — the existing function is reused unchanged, called 25 times.

**Trust:** the `ab_alignment` penalty becomes belief-rate weighted rather than
count-based.

**Tests:** the existing `ab_alignment` tests stay valid (the single-pair function
is untouched). New tests for the distribution wrapper with a deterministic fake
probe function — hermetic, no models, same pattern as the existing probe tests.

**Flowchart: yes, small.** The GRAPH B box gains "×5 probes", and the A-vs-B box
becomes "alignment distribution".

---

# F20 — severity ladder, not weight, is where internal alignment should speak

## Problem

Your question: are these trivial? Right now we cannot tell, because F16's false
positives flood the panel. Once they are gone, what remains is genuinely mixed
and the flat severity ceiling of 2 cannot separate it:

| Failure | Trivial? |
|---|---|
| `quad state 'proximity' != child_2's state 'swimming'` | No. The model contradicts the frozen record about what it saw. |
| `at-risk child_1 used as a threat` | No. Role inversion — the victim named as the hazard. The exact causal-direction error CEE+ exists to catch. |
| `rec 3: remaining_risk duplicates rec 2` | Yes. Severity 0 is correct. |

## Proposed solution

Keep the 0.20 bucket weight. Raise the ceiling: **role inversion becomes
severity 3**, above the current maximum. Cause-and-effect reversal is not the
same class of error as a duplicated field.

## How it solves the problem

The weight answers "how much does internal coherence matter" (0.20 is
defensible). The severity answers "how bad is *this* incoherence", and that is
where the meaning lives. This is F15's severity-blindness one level down.

## What else has to change

**Rules:** severity constants only. **Prompts:** none. **Tests:** the existing
internal-alignment tests assert categories, not severity numbers, so they hold;
add one asserting role inversion outranks a duplicate.
**Flowchart:** no change.

---

# F21 — B_pool: confidently, reproducibly wrong

## Problem

Stage 2 events: `reflect_stopped, rounds: 0, reason: clean, u_before 0.033`.
Zero violations. Near-zero uncertainty. `self_confidence: 0.95`. Five probes
agreed. And the answer was wrong — `child_2` state `swimming`, at risk by
`proximity` — while the caption reads *"another child floats motionless and
unconscious face down"*.

Reflection and petition were **right not to fire**. Every pressure signal we have
is a *self*-consistency signal, and a model that is consistently wrong passes all
of them. This is F5 in its purest observed form.

The unused signal is sitting there. Stage 2 is text-only but it **holds** the
caption, and nothing checks the assessment against it.

## Proposed solution

A caption-grounding check (proposed name S9). Condition words in the caption with
no counterpart anywhere in the record become a trigger, phrased neutrally:

> The caption mentions "unconscious" and "motionless". Your assessment does not
> account for these. Look again.

We quote the caption. We never supply the answer.

## How it solves the problem

- It is the first pressure signal that is **not** self-consistency, so it can
  reach a model that is stably wrong. That is the whole F5 blind spot.
- It routes to a stage-1 petition — the existing route for "the entity list is
  suspect" — with no new machinery.
- It plausibly also recovers D_aerial's missed second hazmat worker, if the
  caption names two.

## What else has to change

**Rules:** new check family S9 alongside S1–S8.

**Prompts:** none new; the trigger text is composed from the caption, same
evidence-quoting pattern reflection already uses.

**Code:** a caption term extractor. This is the risky part — it must not become
a keyword list that smuggles in the answer. Proposed constraint: it may only
flag terms that are **absent** from the record; it may never propose what state
to use. The vocabulary it draws on should be the existing closed state
vocabulary, so it can say "the caption contains a condition word you did not
use" without inventing anything.

**Tests:** caption with a condition word present in the record → no trigger;
absent → trigger; caption empty → no trigger; caption with an off-vocabulary
word → no trigger (guard against the extractor inventing states).

**Flowchart: yes.** New check family S9 on the Stage 2 box, feeding the existing
two-route petition.

---

# Smaller answers

- **"Bound 4/6"** — grounding quality, not a defect count. 4 of 6 entities got a
  box from the DINO detector; 2 fell back to VLM box + SAM. Solid outline =
  detector, dashed = fallback.
- **"a threat in only 0/5 re-asks"** — 0 is literal. The temp-0 answer named
  `car_1` a threat; not one re-ask did. Strongest instability signal we can
  produce, and it should not read the same as 3/5. Proposed: a *singleton claim*
  flag and a per-rec trust floor at 0/5.
- **Both D picks showing target `police_car_1`** — display artifact. Each route
  returns one threat and the panel draws only the first affected object. Should
  list all affected objects and let the routes differ.
- **`spill_1` clicks through to uncertainty only** — zero edges, in no
  recommendation. F18 gives it a real card: *declared hazardous, no target
  named, no recommendation*.
- **"What to intervene on" has no HOW** — keep the two questions apart. Stage 4
  answers *which mechanism is load-bearing*, and `pool_1 · engulfing` is
  defensible for that: remove the water and the drowning stops. That is a
  counterfactual claim, not an instruction to a responder. S6 answers *how* to
  suppress it (inpaint / redact / both). Proposed: relabel the panel
  **"suppression target (for the causal test)"** plus one explaining line.

---

# Change budget, at a glance

| Finding | Prompt | Conformance rules | Alignment formula | Frozen Arm A | Existing tests | Flowchart |
|---|---|---|---|---|---|---|
| F16 dot | 2 edits | **none** | **none** (sanitise input only) | untouched | **none change** | no |
| F17 null case | 1 clause | tolerate empty | none | untouched | none change | yes |
| F18 one-ended | 2 edits | **substantial** | none (fewer edges in) | untouched | **self-loop tests rewritten** | yes |
| F19 probe both | none | none | none (reused ×25) | untouched | none change | yes, small |
| F20 severity | none | constants only | none | untouched | none change | no |
| F21 caption S9 | none | new family | none | untouched | none change | yes |

# Sequencing

F16, F17, F18 and F21 all change what the numbers mean. Tuning trust weights now
would be tuning against a corrupted signal.

Recommended order:

1. **F16 + F17 together.** Both are ours, both are cheap, neither breaks an
   existing test. Re-run all six. F's re-run is the clean experiment on whether
   the safe-scene recommendation was prompt-forced.
2. **F19.** Free on the A side, five calls on the B side, no rule changes.
3. **F18.** The expensive one — rules and tests both move.
4. **F21.** New check family.
5. **Only then** set the trust weights.

Five of the six findings are ours, not the model's. Fixing the interview, not
the witness.
