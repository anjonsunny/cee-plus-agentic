export const meta = {
  name: 'intervention-loop',
  description: 'Full CEE+ intervention reflection loop: (cold) build or (warm) use existing code, then hermetic test+reflect+refine, then live run+reflect-on-live+refine+verify. args: {mode:"cold"|"warm"}',
  phases: [
    { title: 'Build', detail: 'cold only: Builder + Test-author, parallel, independent' },
    { title: 'Test', detail: 'pytest (hermetic eval-for-code)' },
    { title: 'Reflect', detail: '3 critics score Sections A/B' },
    { title: 'Refine', detail: 'apply accepted hermetic findings' },
    { title: 'Live', detail: 'run the live experiment driver (real VLM)' },
    { title: 'Reflect-live', detail: '3 critics score Section C + live-exposed bugs' },
    { title: 'Refine-live', detail: 'apply live-surfaced findings' },
    { title: 'Verify', detail: 're-run pytest, then re-run live to confirm' },
  ],
}

// mode: "cold" generates from the frozen contract; "warm" refines the existing intervention.py.
const mode = (args && args.mode) || 'cold'
const SCENE = (args && args.scene) || 'push_06_drowning_pool'        // live experiment scene
const DIRECTIVES = (args && args.directives) || ''                    // extra refine directives this round
const SCRATCH = '/private/tmp/claude-501/-Users-sunny-Documents-CEE-/c97ec532-4800-4911-b105-0989381f984b/scratchpad'
const DOCS = 'Read INTERVENTION_PLAN.md and INTERVENTION_WORKFLOW.md (plan = intent; workflow = frozen contract + fixed rules + rubric + role prompts).'
const PYTEST = 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate clip_dash && pytest -c tests/pytest.ini tests/test_intervention.py -q'
const DRIVER = `source ~/miniconda3/etc/profile.d/conda.sh && conda activate clip_dash && SCENE=${SCENE} python ${SCRATCH}/run_scene_intervention.py 2>&1 | tail -45`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: { findings: { type: 'array', items: { type: 'object', properties: {
    rubric_id: { type: 'string' }, severity: { type: 'string', enum: ['high', 'med', 'low'] },
    step: { type: 'string' }, problem: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' },
  }, required: ['rubric_id', 'severity', 'problem', 'fix'] } } },
  required: ['findings'],
}

const critique = (lensList, livePhase) => parallel(lensList.map(L => () =>
  agent(`${DOCS} Read intervention.py and tests/test_intervention.py.${L.extra || ''} You are the ${L.lens.toUpperCase()} CRITIC. Rubric slice: ${L.slice}. Walk every item, score pass/minor/major, return STRICT JSON findings (rubric_id, severity, step, problem, evidence, fix). Adversarial; no invented issues.`,
    { label: `critic:${L.lens}`, phase: livePhase, schema: FINDINGS_SCHEMA })))
const collect = (rs) => rs.filter(Boolean).flatMap(r => (r && r.findings) ? r.findings : [])
const refineWith = (accepted, ph, extra = '') => agent(
  `${DOCS} You are the REFINER. Apply these accepted findings to intervention.py and tests/test_intervention.py (minimal change each, no over-refactor); extend tests to lock each fix:\n${JSON.stringify(accepted, null, 2)}\n${extra}\nUse Edit/Write. Return the diffs + state final MOVE_CUTOFF / U-check rule / role logic.`,
  { label: 'refiner', phase: ph })

// ---- Phase 1: BUILD (cold only) ----
if (mode === 'cold') {
  phase('Build')
  await parallel([
    () => agent(`${DOCS} You are the BUILDER. Implement the Builder prompt in INTERVENTION_WORKFLOW.md exactly (frozen contract, all fixed rules, Integration constraints: no top-level import of main, lazy-import helpers, run_counterfactual parses raw VLM JSON, inspect real signatures in main.py). Write intervention.py at the repo root. No tests. Return a one-line summary.`,
      { label: 'builder', phase: 'Build' }),
    () => agent(`${DOCS} You are the TEST-AUTHOR (do NOT read intervention.py). Implement the Test-author prompt in INTERVENTION_WORKFLOW.md: per-step invariants (hermetic, stub vlm_fn, GT via tmp gt_dir), the 2x2 oracle + no-GT case, and the GT-specific leak guard. Inspect the real result schema / .gt.json / conftest import pattern first. Write tests/test_intervention.py. Return a one-line summary.`,
      { label: 'test-author', phase: 'Build' }),
  ])
} else {
  log('Warm start: using the existing intervention.py as the starting point (skipping generation).')
}

// ---- Phase 2-4: hermetic test -> reflect -> refine ----
phase('Test')
const hermetic1 = await agent(`Run and return FULL output, do not edit:\n${PYTEST}`, { label: 'pytest', phase: 'Test' })

phase('Reflect')
const hFindings = collect(await critique([
  { lens: 'implementation', slice: 'Section A (A1-A7) + verify each per-step invariant holds', extra: ` pytest output:\n<<<\n${hermetic1}\n>>>` },
  { lens: 'construct', slice: 'B1,B2,B3,B4,B8,B9', extra: ` pytest output:\n<<<\n${hermetic1}\n>>>` },
  { lens: 'confound', slice: 'B5,B6,B7', extra: ` pytest output:\n<<<\n${hermetic1}\n>>>` },
], 'Reflect'))
const hAccepted = hFindings.filter(f => f.severity === 'high' || f.severity === 'med')
log(`Hermetic critics: ${hFindings.length} finding(s), accepting ${hAccepted.length}.`)
phase('Refine')
const hRefine = hAccepted.length ? await refineWith(hAccepted, 'Refine') : 'no hermetic refine needed'

// ---- Phase 5-7: live -> reflect-on-live -> refine ----
phase('Live')
const live1 = await agent(`Run the live experiment and return FULL stdout, do not edit:\n${DRIVER}`, { label: 'live', phase: 'Live' })

phase('Reflect-live')
const liveExtra = ` Also read the live driver ${SCRATCH}/run_scene_intervention.py and the saved result ${SCRATCH}/${SCENE}_intervention.json. Live stdout:\n<<<\n${live1}\n>>>\nScore your slice AGAINST THE LIVE BEHAVIOR (does the measure hold up on a real stateless VLM?).${DIRECTIVES ? '\nDIRECTIVES for this round (confirm against the live output and ensure they are addressed):\n' + DIRECTIVES : ''}`
const lFindings = collect(await critique([
  { lens: 'confound', slice: 'B5,B6,B7 + Section C (esp. C1 U held under a stateless VLM that does not preserve ids)', extra: liveExtra },
  { lens: 'construct', slice: 'B1,B2,B3,B4,B8,B9 against live behavior', extra: liveExtra },
  { lens: 'implementation', slice: 'Section A + role/arm labeling + core-not-declared path', extra: liveExtra },
], 'Reflect-live'))
const lAccepted = lFindings.filter(f => f.severity === 'high' || f.severity === 'med')
log(`Live critics: ${lFindings.length} finding(s), accepting ${lAccepted.length}.`)
phase('Refine-live')
const lRefine = (lAccepted.length || DIRECTIVES)
  ? await refineWith(lAccepted, 'Refine-live', DIRECTIVES ? ('Round directives that MUST be addressed:\n' + DIRECTIVES) : '')
  : 'no live refine needed'

// ---- Phase 8: re-verify (hermetic + live) ----
phase('Verify')
const hermetic2 = await agent(`Run and return FULL output, do not edit:\n${PYTEST}`, { label: 'verify-hermetic', phase: 'Verify' })
const live2 = await agent(`Run the live experiment with the now-updated code and return FULL stdout, do not edit:\n${DRIVER}`, { label: 'verify-live', phase: 'Verify' })

return {
  mode,
  hermetic_findings: hFindings, hermetic_refine: hRefine,
  live_findings: lFindings, live_refine: lRefine,
  final_hermetic: hermetic2, final_live: live2,
}
