# Week 9 — Mini MLOps: Reproducibility + Tracking

## Goal
Turn the Week 8 EEG pipeline into a reproducible experiment:
- pinned environment,
- config in one place,
- structured run output,
- persistent experiment log.

## Files
- `config.py` — hyperparameters, paths, seeds (single source of truth)
- `run_experiment.py` — loads config, runs EEG pipeline, returns result dict
- `results_log.csv` — append-only log: one row per run (run_id, config, metrics)
- `day1_pin.py` — environment pinning notes/script
- `day2_config.py` — original config extraction (now superseded by `config.py`)
- `day4_track.py`, `day6_refactor.py` — placeholders for tracking/refactoring tasks

## How to run
From repo root (with `.venv` active):
```bash
.venv\Scripts\python.exe week09/run_experiment.py
```
Each run appends a row to `results_log.csv`.

## Change config
Edit `config.py` (e.g., `WINDOW_SIZE = 128` → `256`), save, re-run. New row appears with updated config.

## What we learned
- Reproducibility: same config + same code → same results
- Config separation: no more hard-coded values scattered through code
- Structured output: dict return enables logging, comparison, automation
- Persistent logging: CSV accumulation makes multi-run analysis trivial
- Refactoring: type hints, modular functions, explicit run_id generation

## Experiment log columns
`run_id`, `seed`, `window_size`, `step`, `n_blocks`, `buffer`, `rf_params`, `mlp_params`, `n_windows`, `rf_acc`, `rf_bal_acc`, `mlp_acc`, `mlp_bal_acc`



