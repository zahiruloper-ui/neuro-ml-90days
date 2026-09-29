# Week 9, Day 2 — Config

## What a config is
A single, explicit place (here, a Python file with plain variables) holding every
setting a script needs: seed, data path, hyperparameters, run ID. The training script
reads from this config instead of having values hardcoded/scattered through the code.

## Why hardcoded values are a problem
If hyperparameters are typed directly into the training script, changing an experiment
means editing code and hoping you remember what changed. No permanent record connects
a specific run's results to the exact settings that produced them.

## What we built today
- `config.py` — SEED, DATA_PATH (now a local saved CSV, not a live URL), WINDOW_SIZE,
  STEP, N_BLOCKS, BUFFER, RF_PARAMS, MLP_PARAMS, RUN_ID (timestamp-based).
- Saved a local snapshot of the EEG dataset (`week09/data/eeg-eye-state.csv`) instead
  of loading from a live GitHub URL every run, closing the "data can change silently"
  gap identified on Day 1.
- `run_experiment.py` — imports Week 8's `eeg_pipeline.py` functions (via `sys.path`
  insertion, since `week8` has no `__init__.py`) and imports settings from `config.py`,
  wiring config values into `DatasetConfig`, `SplitConfig`, `ModelConfig` objects.

## Regression test result
Reran the full pipeline through the new config-driven script. Results matched Week 8
exactly: RF mean bal_acc=0.518, MLP mean bal_acc=0.478. This confirms the refactor
changed *structure* (how values are set) without changing *behavior* (what the model
actually does) -- the core goal of safe refactoring.

## Debugging note: import naming
Hit `ModuleNotFoundError: No module named 'config'` because the file was actually named
`day2_config.py`, not `config.py`. Python import statements require the exact filename
(minus .py) to match. Renamed the file to `config.py` to fix it, since this config is
reused across the whole week, not just Day 2.

## The gap this exposes (leads into Day 4)
Config controls what settings produce a run, but nothing yet records which settings
produced which results permanently. Terminal output disappears; nothing on disk links
a RUN_ID to its config and metrics. This is solved by an experiment log (Day 4).