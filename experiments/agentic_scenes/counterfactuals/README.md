# Counterfactual intervention inputs

Manually-edited images used as the **do()** in the CEE+ intervention pipeline (the edited
input a user uploads on the Intervention tab). Each file is a surgical edit of a baseline
scene with one hazard removed or neutralized.

These are **NOT baseline scenes.** They are intervention *inputs*, so they deliberately live
outside `exports/ground_truth/candidates/` — a batch or GT run walks the candidate scene
folders recursively, and a counterfactual must never be scored as if it were a real scene.
They also carry no entry in any `captions.json` manifest for the same reason.

## Contents

- `push_09_tanker_near_fire_no_leak.jpg` — counterfactual of
  `candidates/push_test/push_09_tanker_near_fire.jpg` with the tanker leak removed (the
  "Back to the Future" clean-removal edit of the tanker hazard).
