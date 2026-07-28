# Stage 4 calibration round 1 — reconciled findings and proposal

**Author:** Claude (Fable 5), Claude Code
**Date:** 2026-07-27

**Inputs, both written independently from the same six run records:**

- [`calibration_stage_4_round1_claudecode.md`](calibration_stage_4_round1_claudecode.md) — mine
- [`calibration_stage_4_round1_cowork.md`](calibration_stage_4_round1_cowork.md) — Cowork's

**Method.** I re-derived Cowork's factual claims from the run records and the
code rather than accepting them. Where the two analyses disagree I say which
side wins and why. I have **not** read `..._reconciled_cowork.md` — two
independent reconciliations are worth more than one contaminated by the other,
the same reasoning that makes the two source analyses worth having.

**Status.** Proposal only. Nothing is built. Ledger numbers are proposed, not
written to `FINDINGS.md`.

Runs: A `ui_4b6355e1` · B `ui_e50abbb5` · C `ui_1430cc13` · D `ui_e06a9547` ·
E `ui_6b2d9163` · F `ui_c4c41d47`

---

## 1. Verification of the Cowork assessment

Checked against the run records and the code, not taken on trust.

| Claim | Verdict |
|---|---|
| Summary table (trust / conformance / internal / A-vs-B / recU, all six scenes) | **confirmed — every cell** |
| F_park trust 0.448 < A_fire 0.577 | **confirmed** |
| F_park Graph B empty; b_pick "no hazards present to suppress" | **confirmed** |
| F_park Graph A fabricates `dog_1 running exposes swing_1` | **confirmed** |
| `affected_objects: NON-EMPTY` vs `assumptions_advisory: (may be empty)` | **confirmed** — recommend.py:134 / :139 |
| B_pool raw quad `{"threat":"child_2","state":"proximity"}` | **confirmed** |
| TRUST_WEIGHTS — ab_alignment 0.30, internal 0.20 | **confirmed** — evals4.py:413 |
| No test asserts `_scene_block` | **confirmed** (grep empty) |
| Dot-tolerance test already exists | **confirmed** — `test_recommend.py::..._resolves_dirty_ids_to_frozen` |
| Exactly one self-loop test needs rewriting | **confirmed** — test_evals4.py:133 |
| "all 371 tests" | **wrong — 383 pass** |

One factual error out of eleven checkable claims, and it is cosmetic. The
analysis is sound.

---

## 2. What each analysis contributed

| Finding | Cowork | Claude Code | Whose version to build |
|---|---|---|---|
| `·` separator collision | F16 | P1 | **Cowork's** — traces the failure through frozen `_topological_edge_key` → `resolve_label_class`, and its single `bare_id()` normaliser beats my three scattered strips |
| Null path for safe scenes | F17a | **absent** | **Cowork's** — I missed it entirely |
| Prompt may force the fabrication | F17b | absent | **Cowork's**, but see §4 — the experiment is confounded as designed |
| One-ended causal claims | F18 | P3 + P4 | **Cowork's** — node annotation, not a new edge type; keeps the frozen comparators untouched. Plus my extra ladder rung (§4) |
| A-vs-B unprobed | F19 | P2 | **Cowork's** — it spotted that Graph A ×5 is free |
| Internal alignment severity | F20 | P6 | **Cowork's** — raise the severity ceiling rather than move the bucket weight |
| Caption contradiction | F21 | P5 | **Merge** — Cowork's detector design (S9, extractor constraints) + my three-layer evidence |
| Intervention label | smaller answers | P7 | Same conclusion both sides |
| Duplicate recommendations | — | P8 | Mine only; low priority |
| Graph B right / Graph A wrong | *implicit,×3* | "the one thing I'd underline" | **Promote to a finding** — see F22 |

**Convergence is itself evidence.** Two independent passes landed on the same
three root causes — the `·` collision, unprobed alignment, and the self-loop
family. That makes those diagnoses structural rather than one reader's artefact.

**Cowork is stronger on:** the null path (which I missed), the mechanism trace
through frozen code, and change-budget discipline (what breaks, what doesn't).

**Mine is stronger on:** the three-layer B_pool evidence, layer attribution for
the pathology question, and naming the cross-scene pattern.

---

## 3. Proposed final finding set

| # | Finding | Category | Source |
|---|---|---|---|
| **F16** | The `·` separator carries two meanings; the model copied our notation | B — ours | both |
| **F17** | No null path: signals with nothing to compare score as total disagreement | F — metric defect | Cowork |
| **F18** | One-ended causal claims have no representation (unattached hazard / unattributed victim) | C — lawbook collision | both |
| **F19** | A-vs-B is a point estimate on both sides | F — metric defect | both |
| **F20** | Internal alignment is severity-flat: role inversion scores like a duplicate | F — metric defect | both |
| **F21** | Caption contradiction is invisible to a wholly self-consistent pipeline | **E — model error** + detector gap | both |
| **F22** | Graph B is right where Graph A is wrong | C — and the batch's headline | promoted |
| **O1** | Is the safe-scene recommendation prompt-forced? | **open** | Cowork |

### F22 — the finding neither document filed

Cowork observes this three times (in F16, F17 and F18) and files it as three
defects. It is one phenomenon:

| Scene | Graph B (asked directly) | Graph A (from rec quads) |
|---|---|---|
| B_pool | `pool_1→child_1`, `pool_1→child_2` — **correct** | `child_1→child_1`, `child_2→child_2` |
| F_park | `[]` + *"no hazards present to suppress"* — **correct** | fabricates `dog_1 exposes swing_1` |
| D_aerial | `tanker→spill→hazmat_worker` chain — **correct** | 8 flat edges, all `·proximity` |

Same model, same scene, same run. **Asked "what causes what," the model is
reliably right. Asked to justify a recommendation as a quad, it degrades.**

This reframes the project's own thesis. CEE+ claims recommendations are
declaratively coherent but not causally grounded. The batch says something
sharper: **the model holds the causal structure; the recommendation schema is
where it is lost.** So A-vs-B disagreement is currently a blend of genuine
ungroundedness and our own schema corrupting the A side.

**Consequence: no Stage-4 grounding claim is safe to publish until F16 and F18
land and the two are separated.** Graph B should be documented as the in-run
control — same model, same scene, same run, with the recommendation schema
removed as the only variable.

---

## 4. Conflicts, resolved

**(a) Test count.** Cowork says 371; **383 pass**. Its other test claims all
verified correct, including "zero existing tests need editing" for F16 and
"exactly one self-loop test changes" for F18.

**(b) "Five of the six findings are ours."** By the ledger's own categories:

- **Ours:** F16, F17a, F18, F19, F20 — five, agreed
- **The model's:** F21 — genuinely Category E. We showed it *"motionless and
  unconscious face down"* and it returned `swimming`
- **Undetermined:** F17b, by its own admission, until the paired re-run

Correct count: **four-to-five ours, one model's, one open.** The headline claim
survives without rounding, and the paper is stronger if the count is exact.

**(c) F17b's experiment is confounded.** The proposal adds "recommendations MAY
BE EMPTY" and re-runs F_park to decide whether the fabrication was
prompt-forced. Clean logic, but the clause changes the prompt for **all six**
scenes — a model newly permitted to say nothing may also say *less* on hazard
scenes, so the change that makes F interpretable can move the other five, and
we would then calibrate against a shifted baseline.

**Resolution:** run F_park **paired** — with and without the clause, same
session, everything else fixed. One extra run isolates the prompt effect from
the scene effect. Only after that pair reads clean should the clause enter the
shared prompt.

**(d) F18's disposition ladder has an escape hatch.** It closes at severity 0
when the model answers "none visible." But on a scene where **code has derived
a hazard** — B_pool's `pool_1` medium-bound derivation — a model that says "no
source visible" for a drowning child is contradicting the shared record, not
making an honest declaration.

**Resolution — add a rung:** if a `hazard_bearing` entity exists in the record
and the model still declares no source, that is an **inconsistency**, not a
terminal answer. Without it, "none visible" becomes a way to dodge the question
and score 0 for it.

**(e) F19's cost is understated, and it hides a second question.** "5 extra
model calls" is per scene — **30 per calibration pass** on local Ollama, on top
of existing probes. And probing Graph B at temperature measures something never
yet measured: **is the model's causal belief itself stable?** Today B is the
yardstick, assumed fixed. If B wobbles 5/5, the yardstick is elastic and every
A-vs-B number in the project inherits that. Name it as its own question.

**(f) F21 is stronger than either document has it.** Cowork treats B_pool as a
perception miss that self-consistency cannot catch — true, and S9 is the right
instrument. But the run contains a third layer:

- **Stage 1:** `child_2 · swimming` (`kind=normal`)
- **Stage 2 reason:** *"The child is swimming near an engulfing hazard, which
  **could** lead to a similar situation **if not supervised properly**"* — two
  conditional hedges turning an unconscious drowning victim into a supervision
  concern
- **Stage 4 rank 2:** *"**Rescue the unconscious child** from the water"*, quad
  reaching for `child_2 · distress`

**The model had the information at Stage 4.** So this is not "it could not
see": the caption's content survived to Stage 4 but never entered the record,
and Stage 2 built a rationalisation around the gap.

**Layer attribution, which matters for S5 later:** Stage 1 is a plain
perception error — a bare wrong word, no reasoning, nothing rationalised. The
Stage 2 *reason string* is where the minimisation signature lives, and it sits
exactly where the weak-reason floor already looks, so the detector is nearly
free. Do not label the perception layer a pathology; that over-labelling is
what would make S5 meaningless.

---

## 5. Sequencing

Re-runs are the expensive step (live, needs Ollama), so batch everything
mechanical into one pass.

| Batch | Contents | Why together |
|---|---|---|
| **1** | F16, F17a, F20, F21(a) detector, F18's orphan-hazard prompt line | All cheap, all additive, none breaks an existing test |
| **re-run** | All six + **one extra paired F_park** for O1 | Every alignment number currently held is suspect |
| **2** | F19 (probe both sides) | Free on the A side; 30 calls per pass on the B side |
| **3** | F18 (node flags) | The expensive one — rules and tests both move. **Needs ratification first** |
| **4** | Trust weights | Only after the signals stop being corrupted |

**One change from Cowork's order:** it puts F21 at step 4. Move the detector
into batch 1 — it is additive, touches no existing test, and B_pool is the only
scene where the apparatus is confidently wrong, so the first re-run is the
cheapest chance to see whether it fires. Waiting costs an extra full re-run to
find out.

**Both analyses agree without reservation:** do not tune the trust weights
until F16/F17/F18/F21 have landed. Tuning now would be tuning against a
corrupted signal.

---

## 6. Predictions, to register before the re-run

Neither source document does this. Recording them first turns the re-run from a
look into a test.

| Prediction | Falsifies |
|---|---|
| D_aerial alignment moves off 0.0 | if not, F16 was not the cause |
| F_park trust rises above every hazard scene | F17a's null path |
| B_pool alignment stays low | self-loops survive until F18 |
| B_pool's caption check fires on "unconscious"/"motionless" | F21's detector |
| A_fire and D_aerial internal alignment rise | F16 + F20 |
| F_park paired run: does `[]` appear with the clause? | **O1 — the open question** |

---

## 7. Decisions needed from Sunny

1. **F18's ontology extension.** Node-level flags (`unattached` / `unattributed`)
   instead of edges. Arm A owns the quad ontology and is frozen, so this is an
   Arm B layer recorded alongside Arm A's raw form — the pattern `evals4.py`
   already uses for trust. Needs an explicit yes before build.
2. **The extra ladder rung** for code-derived hazards (§4d).
3. **Whether O1's "MAY BE EMPTY" clause ships** to the shared prompt, after the
   paired run reads clean.
4. **Flowchart edits** (iron rule 9): F17 adds a null branch on the TRUST box;
   F18 adds "attach, or declare unattached/unattributed" to the graph-A build
   box plus a new reflection trigger family; F19 adds "×5 probes" to the GRAPH B
   box; F21 makes Loop 1's check list **P1–P7**.

---

## 8. The one-line summary

Of the seven findings, **five are ours, one is the model's (F21), one is open
(O1)** — and the most important, F22, says that where we can compare, the model
knows the causal structure and our recommendation schema loses it. Fixing the
interview, not the witness, holds for this batch too — but F21 is a real
reminder that the witness has genuine deficits, and that every pressure signal
we currently own is a self-consistency signal, blind to a model that is
confidently and reproducibly wrong.
