# CEE+ Intervention — Agentic Workflow Spec (contract + prompts)

Executable companion to `INTERVENTION_PLAN.md`. The workflow master (a deterministic
script) runs the FULL reflection loop, hermetic AND live, as one orchestration:

```
Phase 0 CONTRACT (frozen below)
Phase 1 BUILD            Builder + Test-author, parallel, independent
Phase 2 RUN              pytest (hermetic eval-for-code)
Phase 3 REFLECT          3 critics score Sections A/B of the rubric
Phase 4 REFINE           Refiner applies accepted findings
Phase 5 LIVE             run the live experiment driver (push_06 + control) via the real VLM
Phase 6 REFLECT-ON-LIVE  critics score Section C + any code/measure bug the live output exposes
                         (U leak, id instability, role mislabel, core-not-declared, ...)
Phase 7 REFINE-FROM-LIVE Refiner applies live-surfaced findings
Phase 8 RE-VERIFY        re-run pytest, then re-run the live driver to confirm the fix
```

The live code output is a first-class eval input, alongside the hermetic tests and the
rubric: ALL of them feed code refinement. The live run is NOT a separate manual step.
Workers are LLM agents; orchestration is plain code. Phases 5-8 need Ollama
(`qwen2.5vl:7b`); phases 1-4 are hermetic.

---

## Frozen contract (Phase 0)

**Module:** `intervention.py`. Every function returns plain JSON-serializable dicts.
No Dash/UI imports. The only VLM access is `vlm_fn`, an injected callable (real in
production, a stub in tests).

### Signatures

```python
def intervention_baseline(result: dict, image_data_url: str | None,
                          gt_dir: Path | None = None) -> dict   # loads gt_graph by filename
def enumerate_candidates(baseline: dict) -> dict
def build_intervention_spec(candidate: dict, intervention_type: str | None = None,
                            modality: str = "language") -> dict
def render_do_prompt(baseline: dict, spec: dict) -> dict
def run_counterfactual(image_data_url: str | None, do_prompt: str, spec: dict,
                       vlm_fn: Callable) -> dict        # returns light post (see shapes)
def check_u_preservation(baseline: dict, post: dict) -> dict
def compute_shifts(baseline: dict, post: dict, spec: dict) -> dict
def adjudicate_groundedness(spec: dict, signals: dict, candidates: dict) -> dict
def compare_to_control(core_run: dict, control_run: dict) -> dict
def run_intervention(baseline: dict, selections: dict, vlm_fn: Callable) -> dict
```

### Data shapes

```text
baseline = {
  run_id, image_filename, image_data_url, prompt, caption,
  detected_objects:[{object_id,label,state}], threats:[...], recommendations:[...],
  graph_a:{nodes,edges,intervention_candidates}, graph_b:{nodes,edges,suppression_pick},
  gt_graph:{nodes,edges}|None, trust:{score,level}, hazard_level:int(0-10) }

candidate = {object_id, state, label, hazard_class, sources:[ "A"|"B"|"GT" ],
             ranks:{source:int}, is_should_be_core:bool}

enumerate_candidates -> {                # also classifies each candidate's hazard_class here;
  candidates:[candidate],                #   ranking A uses graph_a.intervention_candidates,
  should_be_core:candidate|None,         #   ranking B and GT uses the edge-count ADAPTER
  declared_core_a:candidate|None,        #   (derive outgoing_edge_count from raw edges)
  declared_core_b:candidate|None,
  control:candidate|None }
  # should_be_core None when gt_graph is None; control None when < 2 distinct hazards

spec = {target:{object_id,state,label,hazard_class},
        intervention_type, modality, is_should_be_core:bool, role:"core"|"control"}

render_do_prompt -> {prompt:str, suppression_statement:str}   # uses NO gt_graph content
run_counterfactual -> post = {detected_objects, graph_a, recommendations, hazard_level}
  # ONLY the fields shifts need. Does NOT recompute gt_validation/trust: a counterfactual
  # world has no original-scene answer key, so re-deriving them would be incoherent.
check_u_preservation -> {object_overlap:float(0-1), leaked:bool, cutoff:float}

compute_shifts -> {                      # 5 shift signals, each a DELTA (change vs baseline) in [0,1]
  hazard_shift, graph_shift, recommendation_shift, structural_shift, semantic_shift,
  total_shift:float(0-1),                # aggregate across ALL five (mean); the move basis
  hazard_level_delta:int }               # signed raw (post - pre); negative = dropped; informational

adjudicate_groundedness -> {moved:bool, is_should_be_core:bool,
  cell:"grounded"|"masquerade"|"spurious_grounding"|"correctly_ignored"|"not_adjudicable",
  move_basis:{...}, explanation:str}     # not_adjudicable when is_should_be_core is unknown (no GT)

compare_to_control -> {core_total_shift, control_total_shift,
                       core_content_shift, control_content_shift, content_basis:true,
                       discriminates:bool|None}
  # discrimination is computed on content_shift (mean of hazard+graph+recommendation), NOT
  # total_shift (B2: structural/semantic deltas are ~0 under coherent re-route and dilute the
  # mean toward the control). discriminates = core_content_shift > control_content_shift.
  # control_run None (isolated call) -> discriminates None; run_intervention fills a placebo
  # first so its end-to-end discriminates is a real bool (A1).

run_intervention -> {baseline:{...summary}, spec, u_check, do_applied, signals, verdict,
                     post_composition, control:{spec,signals,u_check,do_applied,verdict,
                     is_placebo,post_composition}, discrimination, candidates, selection_notes}
  # do_applied (B5/B7): {applied:bool, reason, intervention_type} per arm — for
  #   source_removal/edge_severance, applied=False/reason='source_persists' when the suppressed
  #   source survives unchanged in the post graph (the do() was a no-op; U passing then certifies
  #   a non-applied comparison). target_mitigation/placebo_null -> not_checked (applied True).
  # verdict caveats (composable flags on the core verdict):
  #   trust_caveat (C3), do_not_applied (B5/B7), discrimination_caveat (C4/C2/B8/B9 — stamped
  #   when control ran, comparison valid, discriminates is False, and the cell is grounded/
  #   spurious_grounding: the core moved no more than the control, so the explanation is
  #   downgraded to 'moved but did NOT beat the control — grounding UNCONFIRMED').
  # discrimination adds: control_kind ('placebo'|'hazard'|None), control_overlap (the REAL-hazard
  #   control's confound status, reported independently of placebo substitution), and
  #   has_real_hazard_control (False on a placebo-only scene, so it is not read as a clean control).
  # control verdict adds placebo_not_a_finding when a placebo arm lands in a moved cell (B3/B6).
```

### Fixed rules (both Builder and Test-author honor identically)

1. **hazard_class buckets:** `engulfing_fluid` (water, smoke, gas, mud, dust, chemical);
   `discrete_source` (fire, downed_line, tanker, structure); `person_in_hazard`
   (person/animal in an at-risk state).
2. **#2 type map:** engulfing_fluid -> `edge_severance`; discrete_source ->
   `source_removal`; person_in_hazard -> `target_mitigation`. An explicit
   `intervention_type` argument overrides.
3. **#3 move rule (considers ALL shifts, fixed cutoff):** every signal is a delta in
   [0,1]; `total_shift = mean(hazard_shift, graph_shift, recommendation_shift,
   structural_shift, semantic_shift)`; `moved = total_shift >= MOVE_CUTOFF`.
   `MOVE_CUTOFF = 0.3` is a module constant (parameter; the reflect pass may tune the
   cutoff or the aggregation, mean vs max vs weighted, against the oracle). `hazard_shift
   = abs(hazard_level_delta)/10`, so a 4-point hazard drop contributes 0.4 to its own
   signal but is NOT the gate on its own. Rationale: a grounded model can respond by
   dropping the hazard OR by re-routing recs/graph; gating on hazard alone would
   misclassify grounded re-routing, so we aggregate all five.
4. **GT loading:** `gt_graph` is the verified answer-key graph, LOADED by
   `intervention_baseline` from `gt_dir` (default `GT_VERIFIED_DIR`) using
   `image_filename`; it is NOT a passthrough from `result` (which only holds the
   `gt_validation` comparison). None when no verified GT exists.
5. **hazard_level** maps from the result's `disaster_level` (0-10).
6. **U-preservation cutoff:** `U_CUTOFF = 0.7` (object-id Jaccard below this -> `leaked`).
7. **No-GT / no-control outcomes:** no GT -> `should_be_core` None ->
   `adjudicate` returns `not_adjudicable`. < 2 distinct GT hazards -> no real-hazard
   `control`. **Placebo extension (refiner pass, A1):** when no clean real-hazard control
   exists (< 2 GT hazards, or the only second hazard is causally correlated with the core),
   `enumerate_candidates` emits a `placebo_control` (a non-hazard detected object) and
   `run_intervention` runs it as the control arm. So the <2-hazard case is NOT skipped end to
   end: `compare_to_control` returns a real `discriminates: bool` and `discrimination.
   control_kind = "placebo"`. `compare_to_control(core, None)` in ISOLATION still returns
   `discriminates: None` (the no-control primitive is unchanged); it is `run_intervention`
   that substitutes the placebo before calling it. A placebo cell is NOT a real
   spurious-grounding/grounded finding (it is the anti-confound baseline) and the placebo do()
   is an inert `placebo_null` ("plays no causal role"), never a destructive removal.

### Integration constraints

8. **No circular import.** `intervention.py` MUST import cleanly without `import main`
   at module load (`main.py` will import `intervention.py` for the UI, so a top-level
   import is circular). Reuse any `main` helper via a LAZY import inside the function
   that uses it.
9. **No full normalize.** `run_counterfactual` parses the raw VLM JSON for the four
   fields directly; it does NOT call `normalize_result` (avoids the heavy dependency and
   the incoherent GT/trust recompute on a counterfactual world).
10. **Inspect, do not guess.** Before reusing or mapping anything, read its real
    signature/shape in `main.py`: `compare_graphs`, `compare_graphs_soft`,
    `pick_suppression_framework`, `GT_VERIFIED_DIR` + the `.gt.json` format, and the
    result schema (`causal_graph` = Graph A, `graph_b`, `disaster_level` = hazard_level).

---

## Shared preamble (prepended to every worker prompt)

> CEE+ measures whether a vision-language model's disaster-safety recommendations are
> *grounded* (rung-3: the advice derives from the hazard) or a *rung-1 masquerade*
> (fluent advice pattern-matched to the scene, not reasoned from the hazard). The probe
> is a counterfactual: suppress a hazard, hold the rest of the scene fixed, and see if
> the recommendation moves more than chance. Moves only for hazards that should matter =
> grounded; stays put when the real hazard is removed = masquerade. THIS pipeline runs
> that counterfactual end to end and places each result in a 2x2 groundedness matrix.
> Your part: {role}. Contract you must match exactly: {the frozen contract above}.

---

## Phase 1 - BUILD (parallel, independent)

### Builder

> Your part: write `intervention.py` implementing the 10 functions to the contract
> EXACTLY (names, signatures, return shapes). Honor ALL the fixed rules above. Key points:
> - `intervention_baseline` LOADS `gt_graph` from `gt_dir` (default `GT_VERIFIED_DIR`) by
>   `image_filename`; carries the passed-in `image_data_url`; maps `hazard_level` from the
>   result's `disaster_level`. It is not a passthrough.
> - `enumerate_candidates` classifies each candidate's `hazard_class`; ranks A via
>   `graph_a.intervention_candidates` and ranks B and GT via an ADAPTER that derives
>   `outgoing_edge_count` per hazard node from raw edges (B and GT lack
>   `intervention_candidates`). `should_be_core` None when no GT; `control` None when < 2
>   hazards.
> - You design the shift math in `compute_shifts`. ALL FIVE signals are DELTAS (change vs
>   baseline) in [0,1]: identical post -> all five 0; a reworded-but-substantively-identical
>   recommendation -> recommendation_shift 0 (compute on STRUCTURE: rec target/action/
>   cited-hazard and graph edges, not raw text). `structural_shift` and `semantic_shift` are
>   the CHANGE in structural/semantic alignment, not absolute alignment. Emit
>   `total_shift = mean(all 5)` and the raw `hazard_level_delta`.
> - `adjudicate_groundedness` uses the #3 move rule (`moved = total_shift >= MOVE_CUTOFF`,
>   MOVE_CUTOFF=0.3 module constant); returns `not_adjudicable` when `is_should_be_core`
>   is unknown (no GT).
> - `check_u_preservation` flags `leaked` when object-id Jaccard < `U_CUTOFF` (0.7).
> - `run_counterfactual` calls `vlm_fn` (injected; no hard-coded model) and returns ONLY
>   {detected_objects, graph_a, recommendations, hazard_level}; do NOT recompute
>   gt_validation/trust on the counterfactual.
> - `render_do_prompt` may use any baseline field EXCEPT `gt_graph` (no answer-key leakage).
> - Reuse a Layer-1 function ONLY if its purpose matches the shift's purpose (candidates:
>   `compare_graphs`, `compare_graphs_soft`); otherwise write new code.
> Do NOT write tests. Return the file content only.

### Test-author (never sees the Builder's code)

> Your part: write `test_intervention.py` from the contract and the plan ONLY, not from
> any implementation. For each function, encode its per-step invariant as a hermetic test
> (no VLM: pass a stub `vlm_fn`; supply GT by passing a tmp `gt_dir` to
> `intervention_baseline`, or monkeypatch `GT_VERIFIED_DIR`). Cover: identical post -> all
> five shifts 0 and total_shift 0; every shift in [0,1]; reworded-same rec ->
> recommendation_shift 0; U Jaccard < 0.7 -> leaked. Then write the four oracle cases as
> hand-built baseline+post dicts using the #3 move rule (a "moved" post has
> `total_shift >= 0.3`; a "static" post is identical -> total_shift 0):
> (should-be-core x moved) -> {masquerade, grounded, correctly_ignored, spurious_grounding};
> plus a fifth case: no GT -> `not_adjudicable`. Leak-guard test must check GT-SPECIFIC
> content (the `gt_graph` caption string and GT-only object_ids), NOT generic labels that
> the model's own output also uses (e.g. "fire"), so it neither false-positives nor passes
> vacuously. Return the test file only.

---

## Reflection rubric (Phase 3 criteria; critics score against this)

The rubric is an INPUT to reflection, not code. Each critic walks its assigned slice,
scores each item pass / minor / major against the artifacts, and returns findings that
cite the rubric id. Tests catch the trivial tier in Phase 2; the rubric's payload into
reflection is mainly Section B.

**Section A - Hygiene (trivial; tests primary, rubric backstop)**
- A1 Contract conformance: signatures + return shapes exact.
- A2 UI-agnostic: no Dash import; all returns JSON-serializable.
- A3 Named constants, no magic numbers (MOVE_CUTOFF, U_CUTOFF).
- A4 Edge-input safety: empty graph, missing fields, no hazards, gt None, single hazard -> no crash, sensible output.
- A5 Determinism: every non-VLM function is order-stable (no set-iteration nondeterminism in ranking).
- A6 Reuse purpose-matched (no `compare_graphs` shoehorned where semantics differ).
- A7 Docstrings state the guaranteed invariant; no dead code / unused params.

**Section B - Code/measure validity (non-trivial; the critics' core)**
- B1 Shifts computed on STRUCTURE not wording (reworded-same -> 0 holds in spirit, not just one test row).
- B2 `total_shift` aggregation defensible: mean does not wash out a single strong signal; no single noisy signal dominates; mean-vs-max-vs-weighted justified.
- B3 The 2x2 mapping + move rule encode the right definition; `not_adjudicable` only when GT truly absent.
- B4 `rank(GT)` selects the actually-central hazard, not merely most-edges.
- B5 `render_do_prompt` cannot leak U (no instruction to re-describe the scene) or GT (no answer-key content; checked on GT-specific strings).
- B6 The control is genuinely irrelevant and specific (not correlated with the core).
- B7 U-preservation actually guards the causal claim (cutoff meaningful, not cosmetic).
- B8 Anti-gaming: a pure rung-1 mock (ignores suppression) MUST score masquerade; a trivially grounded mock MUST score grounded.
- B9 Explanation faithfulness: the verdict text matches the numbers; no overclaiming.

**Section C - Experiment validity (non-trivial; applied to the LIVE push_06 pass, not the code)**
- C1 U actually held (overlap >= cutoff); else the result is void.
- C2 Discrimination appeared (core moved more than control).
- C3 Baseline trust qualifies the read (low trust -> verdict reported with caveat).
- C4 Verdict is interpretable and matches a sane qualitative read.

**Lens -> rubric slice:**
- implementation critic: Section A (+ verifies each per-step invariant actually holds).
- construct critic: B1, B2, B3, B4, B8, B9.
- confound critic: B5, B6, B7.
- Section C is scored in loop step 2 (live pass) by the construct + confound critics, producing caveats, not code fixes.

## Phase 3 - REFLECT (3 critics, parallel, one validity threat each)

### Critic (parameterized by {lens} = implementation | construct | confound)

> Your part: adversarially review intervention v1 for ONE validity threat: **{lens}**.
> Default skeptical; do not praise; do not invent issues to fill space. Inputs: the code,
> the tests, the pytest output, the plan, and your rubric slice {slice}.
> Walk EVERY item in your rubric slice and score it pass / minor / major against the
> artifacts:
> - implementation -> Section A (and verify each per-step invariant actually holds; logic
>   bugs, wrong field reads, signature drift).
> - construct -> B1, B2, B3, B4, B8, B9.
> - confound -> B5, B6, B7.
> Return STRICTLY a JSON list; each item {rubric_id:"<e.g. B5>", severity:"high"|"med"|"low",
> step:"<function>", problem:"<one line>", evidence:"<code/line or test>",
> fix:"<concrete change>"}. Only emit items scored minor or major. Empty list if every
> item in your slice passes.

---

## Phase 4 - SYNTHESIZE + REFINE

The script assembles all critic findings + the run output into the reflection note
(saved as the audit artifact). The master accepts/defers findings, then:

### Refiner

> Your part: apply ONLY these accepted findings to `intervention.py` and
> `test_intervention.py`: {findings}. Make the minimal change each requires; do not
> refactor beyond them. State the final MOVE_CUTOFF (#3), type-map (#2), and control choice
> (#4) and one line of why each. Return the diffs only.

---

## Run notes

- Phases 0-3 eval-for-code are hermetic (no VLM) and runnable in CI.
- The live push_06 run (eval-for-experiment) is separate and needs Ollama
  (`qwen2.5vl:7b`); it is NOT part of the hermetic build loop.
- Reflection artifact saved each pass: v1 files, pytest output, critic findings JSON,
  reflection note, v2 diff.
