# Week 9, Day 3 — Run

## Goal of the day
Turn `run_experiment.py` into a script that returns structured results (a dictionary),
not just printed text, so results become usable by other code (specifically tomorrow's
logging step) instead of disappearing into the terminal.

## What changed from Day 2
`main()` now builds and returns a dict containing: run_id, seed, all config values used
(window_size, step, n_blocks, buffer, rf_params, mlp_params), and the resulting metrics
(rf_acc, rf_bal_acc, mlp_acc, mlp_bal_acc). Printing became a separate, secondary step
in the `if __name__ == "__main__":` block. This is a common pattern: separate
"compute the result" from "display the result."

## Precision vs. real difference
Day 2's printed output showed `bal_acc=0.518` (3 decimals via f-string formatting).
Day 3's dict-based output showed `bal_acc=0.5179` (4 decimals via explicit rounding).
Same underlying number -- always check whether an apparent difference in output is a
real behavior change or just a display/precision difference before assuming something
broke.

## Experiment: changing WINDOW_SIZE (128 -> 256)
- n_windows barely changed (233 -> 231), because window *count* is controlled mainly by
  `step`, not `window_size` (confirmed by rereading the sliding-window loop logic).
- Metrics did shift: RF bal_acc 0.518 -> 0.507 (slightly worse), MLP bal_acc
  0.478 -> 0.480 (slightly better). Small but real changes, confirming config values
  are genuinely wired into pipeline behavior.

## Why window size affects results despite same raw data
Bigger windows average mean/std features over more time samples, which can blur
together different underlying states (e.g. eye-open vs eye-closed) that a smaller
window would keep separate. But bigger windows also smooth out short-term signal
noise. This is a real tradeoff, not simply "bigger is better" or "smaller is better" --
depends on how fast the underlying states change vs. how noisy the raw signal is.

## The limitation this exposes (leads into Day 4)
Manually running multiple experiments (e.g. 6 different window sizes) and comparing
by eye breaks down fast: repeated manual file edits (error-prone), scrolling through
separate terminal outputs one at a time, and no permanent link between a specific
run's config and its results. This motivates building an experiment log that appends
a structured row per run.