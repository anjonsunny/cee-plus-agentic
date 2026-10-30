# UI cleanup plan: clean view and detailed view

Written by a planning agent on a worktree copy (2026-10-30), reviewed and
filed here. Nothing in it is built yet.

## 0. How it works

- Two views of the same run. Detailed is today's screen, unchanged. Clean
  puts the verdict, the recommendations, pathology tickets, the causal
  chain and trust on top; some panels fold shut; some appear only in
  detailed view.
- Nothing in clean view hides silently (iron rule 8): every folded section
  shows a one-line gist in its header; every detailed-only panel that has
  content is named in a footer: "hidden in clean view: ...".
- The existing renderers stay as they are. Each gets a `mode="detailed"`
  keyword; clean view is new, small builders in front of them. With the
  default mode the output is identical to today, so today's tests pass
  untouched.

## 1. Inventory

Key: S = shown in clean view. F = folded shut, with a one-line gist.
D = detailed view only.

Top of page
- `instruments_component` tiles: S for four (entities, open tickets,
  bucket, trust); D for the rest.
- `scene_component` (image, boxes, pins, ribbon): S.
- `inspector_component` modal: S.
- Scrub slider: D (hidden by CSS; clean view follows the live run).
- Agent dock (ASK THE ANALYST): S.
- The H1 says "STAGES 1-2" — out of date; fix in step 1.

Stage 1
- `rail_component`: F. Gist: "6/6 steps done · 9 found · 2 fixed, 1 stood".
  A failed station stays S as a red line.
- `tickets_component`: F as count chips; open amber tickets stay S as chips.
- `rag_shadow_panel`: D.
- BEFORE PETITION block: D; clean view shows one purple line
  "re-perceived after a petition".

Stage 2 (`assess_component`)
- Verdict banner: S.
- Threats / at-risk lists: S as red and orange chips; per-entity vote
  badges: D.
- DERIVED HAZARD notes: S (code changed the evidence; cannot hide).
- Ghost entities: D, counted in the footer.
- Uncertainty: S as one chip "measured: 0.10 unsure"; self-reported value
  and calibration gap: D. The `unc-view` toggle: D.
- Assessment tickets: F as count chips; open ones show their kind.
- Reflection ledger: F. Gist: "reflection: 2 rounds · verdict unchanged".
- Petition panel: F; the FINAL DECISION line is S.
- Judges roster, RAG shadow, GT overlay: D.

Stage 4 (`stage4_component` + `_assemble_stage4_sections`)

| key | today | clean |
|---|---|---|
| verdict | full trust panel | S as a compact card: score + band + bar, "what to do", top 2 pull-downs, holds-up line. Minor factors, per-rec list, footnote: D |
| recs (+picks) | full rec cards, across-recs, advisory | S as compact cards: rank, action, trust chip, consequence chip, red chip if a pathology ticket names the rec; reason/quad/rest fold inside. Across-recs: F. Thinking, picks, advisory: D |
| stability | uncertainty panels | F. Gist: "0.12 unsure over 5 re-asks · Graph B 0.30 unsure" |
| graphs | A/B views + conformance | F. Gist: "Graph A 7 links · Graph B 6 links · well-formed / n breaks" |
| alignment | full | S: verdict sentence, gate warning, two plain bars. Resolved ids, decomposition, effect toggle: D |
| bench | judge cards | D. Footer: "judges' bench: off this run" or "n cards" |
| register | counts, chain, rule tickets, pathology tickets, probe folds | S: pathology tickets (with their chain) and the run's chain. F: rule tickets as an amber count chip. D: probe answers / thinking / who looked |
| live view | 15-row step list + early pathology | S: "now: <step> · n of 15" plus early tickets; full list F |
| error state | error line | S |

## 2. Clean-view layout

```
[ drop image ][ caption.............. ][ANALYZE][ replay ]   view: (*) clean ( ) detailed
> run settings: LangGraph · qwen3.8:27b · rule lookup both · no extra probes
[ 9 entities ][ 0 open ][ SEVERE ][ trust 0.62 ]
+- scene with boxes + pins -+   STAGE 1 · PERCEPTION  done 6/6
|                           |   9 found · 2 fixed · 1 stood   > how perception went
+---------------------------+   STAGE 2 · SCENE ASSESSMENT  done
 ask the analyst [.......]      [ DISASTER · fire · level 8 · SEVERE ]
                                threats: (fire_1) (smoke_1)   at risk: (man_1)
                                measured: 0.10 unsure  > 2 tickets  > reflection: 1 round
                                STAGE 4 · RECOMMENDATION  done
                                +- TRUST 0.62 · moderate  #######...  -------------+
                                | what to do: review the flagged recommendations   |
                                | pulls down: acts on a danger it doesn't hold     |
                                | holds up: stable on re-ask · targets agree       |
                                +--------------------------------------------------+
                                #1 Evacuate man_1 upwind   trust 0.81 · high   > why
                                #2 Close the valves        trust 0.44 · med   [! concealment]
                                TICKETS  [2 pathology]  [> 4 rule]
                                +- PATHOLOGY · concealment · specialist delayed  OPEN -+
                                | no longer acts on tanker_truck_1                      |
                                | goals o motive o condition o recs o pathology . ground|
                                +-------------------------------------------------------+
                                THE RUN'S CHAIN  o--o--o--o--.--.
                                ALIGNMENT: the advice and its own beliefs diverge
                                  backed by its belief ######..  acts on what it sees ####...
                                > 3 · STABILITY — 0.12 unsure over 5 re-asks
                                > 4 · THE CAUSAL GRAPHS — A 7 links · B 6 links
                                hidden in clean view: judges' bench (off) · suppression
                                  target · assumptions advisory · 2 ghost entities
```

Visual rules: existing palette only (amber rule tickets, red pathology,
purple judges/petitions, green benign, slate for the rest) in module-level
constants; one coloured-border card per stage (the verdict), everything
else a white card with the `--line` border; chips reuse `_make_chipper`;
the chain in clean view is a horizontal stepper (`_chain(s4, tk,
layout="row")`, dots + short labels, detail on hover); detailed view keeps
the vertical chain. Colour clash to settle: the chain paints motive and
condition `#7F77DD`, close to the judges' purple (question 4).

## 3. The switch

- Where: `dcc.RadioItems(id="view-mode")` in the essentials row of the
  control strip.
- Storage: Dash `persistence=True, persistence_type="local"` on that
  RadioItems (localStorage; page-level, not per run, not in events.jsonl).
- Page-level CSS: one clientside callback sets the outer `wrap` div's
  className to `wrap mode-clean` / `wrap mode-detailed`;
  `.mode-clean .only-detailed{display:none}` hides layout-level pieces
  (`unc-view` toggle, scrub slider) with no server round trip.
- Renderers: add `Input("view-mode","value")` to `render` after
  `unc-view`; pass `mode` to rail / tickets / instruments / assess /
  stage4; each starts `if mode == "clean": return _x_clean(...)`; in
  `stage4_component`, mode reaches `_assemble_stage4_sections`.
- The render callback stays at 15 outputs; mode only changes what fills
  them.
- Render cache: append `mode` to the END of the cache key (`render` reads
  `prev_key[6]` as assess_status; inserting earlier breaks auto-collapse).
  Switching views re-renders once and resets folds — fine for a
  deliberate switch.
- `«sec:»` markers: give `_S4_SECTIONS` a fifth field `clean` =
  "show" | "fold" | "hide". Partitioning unchanged; "show" swaps the body
  for the clean builder; "fold" renders closed with `_section_gist(key,
  s4)` in the Summary; "hide" drops the section and names it in the
  footer. Pull `_ticket` / `_pathology_ticket` out of `_register_panel` to
  module level so the clean register reuses them.

## 4. Control strip

- Always visible: upload, caption, ANALYZE, replay, view switch.
- Behind `html.Details(className="run-settings", open=False)`: the Summary
  shows the current settings as chips from a pure `settings_summary(...)`
  (one small callback: 8 toggle values in, the Summary's children out); a
  slow choice gets a warning chip ("judges on · about 1 hour", "27B
  subject · about 3x slower"); inside, two rows — "how it runs" (control,
  subject, rule lookup) and "extra probes (each adds time)" (stage 4
  judges, re-asks, concealment, concealment · environment, sandbagging).
- All IDs and defaults stay the same, so `start_run` and its States are
  untouched.

## 5. Build order (small PRs; each ships hermetic tests; main.py untouched;
no flowchart edit — UI only)

1. Control-strip tidy (~90 lines, low risk). Tests: all 8 toggle IDs still
   in the layout with today's defaults, inside `run-settings` open=False;
   `settings_summary` defaults read "no extra probes", judges on contains
   "about 1 hour".
2. View switch plumbing, no visual change (~60 lines, low-medium risk:
   callback parameter order). Tests: `view-mode` exists with
   persistence_type local; for every event prefix,
   `component(d) == component(d, mode="detailed")` for all five renderers;
   `render` inputs include view-mode; mode is the last key element.
3. Stage 4 clean view (~250 lines, medium). Policy field, clean verdict
   card, compact rec cards, `_section_gist`, hidden-note footer. Tests on
   `_s4_sectioned()`: contains trust score/band, every rec action, every
   pathology summary, "THE RUN'S CHAIN"; does not contain "SUPPRESSION
   TARGET", "ASSUMPTIONS ADVISORY", bench text; folded sections open=False
   with gist; footer names every hidden section with content; empty,
   error and live states don't crash at any prefix.
4. Horizontal chain (~60 lines, low). Tests: six step titles identical in
   both layouts; last dot dashed.
5. Stage 2 clean view (~150 lines, medium); BEFORE PETITION wrapper
   becomes a one-line badge. Tests: verdict banner, threat/at-risk ids,
   DERIVED HAZARD present; RAG shadow and judges roster absent; reflection
   fold shows round count; petition FINAL DECISION visible.
6. Stage 1 clean view + trimmed instruments (~120 lines, low-medium).
   Tests: summary counts match `d`; open tickets as amber chips; the
   `run_error` fixture shows a red failed line; 4 tiles clean / all tiles
   detailed.
7. Stage 4 live view in clean mode (~50 lines, low). Tests: "now: ... ·
   n of 15"; early pathology tickets still shown.
8. Tidy (~40 lines): colour constants, H1 wording, `_assemble_stage4_sections`
   docstring ("five" -> seven), rename `test_stage4_renders_five_numbered_sections`.

Total about 820 lines, about 40 new tests. Steps 1, 2, 4 are independent;
3, 5, 6 need 2.

## 6. Questions for Sunny (before step 1)

1. One switch for the whole page, or one per stage card? (Built for one;
   per-card is a small change later.)
2. Which view does a fresh browser open in, clean or detailed?
3. Rule tickets in clean view: folded behind a count chip, or shown in
   full like pathology tickets?
4. The chain paints motive and condition purple, close to the judges'
   purple. Change those two to slate or blue so purple means judges only?
5. During a live run, clean view shows only "now: <step> · n of 15", or
   the full step list?

## 7. Decisions so far (Sunny, 2026-10-30)

- The clean-view builders go in a separate file, `agentic/ui_clean.py`,
  imported by `ui.py`; the switch itself lives in `ui.py`.
- UI cleanup is developed on its own branch off `agentic` and merged step
  by step; pathology work continues on `agentic` in parallel.
- The space below the image, reserved for the conversation agent, goes
  to the Stage 4 card in clean view; the agent dock shrinks to one input
  line with the transcript opening only when there is one. Detailed view
  keeps today's layout.

## 8. Answers to the five questions (Sunny, 2026-10-30)

1. One switch for the whole page.
2. A fresh browser opens in clean view.
3. Rule tickets in clean view: a count chip.
4. The chain's motive and condition dots change to slate, so purple
   means judges only.
5. Live run in clean view: "now: <step> · n of 15" with the early
   pathology tickets; the full step list folded (my choice — the short
   line is what the clean view is for, and the fold keeps the detail one
   click away).
