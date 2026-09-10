# Honest coupled rounds of 2026-09-10 (Option B build, 27B, development seeds)

Where things are, so nothing is lost:

- `grades/honest<N>_grades.json_seed<seed>.json` -- grade_round.py output per seed, rounds 2-5
  (round 6 is added when graded). Summarise with `python launch/summ_grades.py grades/honest<N>_grades.json`.
- `launch/rerun_honest<N>.sh` -- the exact launch command of each round (keys must be sealed d---------;
  the custody preflight refuses readable worked answers, including pytest scratch dirs).
- The run directories themselves stay under `campaign3_blind/runs/C*_27b_MCP_seed<seed>` (rounds 2-6 seeds:
  C3 5561-63/5571-73/5581-83/5591-93/5611-13, C1 6211-13/6221-23/6231-33/6241-43/6261-63,
  C2 6311-13/6321-23/6331-33/6341-43/6361-63).
- Per-step OpenRouter trials (one step of a coupled run at a time, validated by running the code):
  `campaign3_blind/step_trials/`. Results land there as `results_<tag>.json` and `deck_<tag>_<i>.yaml`.
- HANDOFF.md section 5 carries the round table and the measured causes.

A backup of this directory and of step_trials is copied to /media/alexander/PortableSSD/oasis_honest_rounds_20260910/
(exFAT: no permission bits, so never put answer keys or worked-answer fixtures there).
