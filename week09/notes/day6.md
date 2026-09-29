# Week 9, Day 6 — Refactor

## Goal of the day
Improve code organization in run_experiment.py by breaking the monolithic main() into
smaller, focused functions with type hints, without changing experiment behavior.

## What changed
- Renamed main() → run_experiment() with explicit type hints and a docstring
- Added type hints to log_result() (result: dict[str, object], log_path: str | None)
- Moved run_id generation inside run_experiment() (tied to experiment start time, not
  config module import time)
- Removed RUN_ID from config.py (no longer needed, was evaluated at import time)
- Cleaned up imports (removed unused RUN_ID import)

## Why move run_id generation into the function
Previously RUN_ID = datetime.now().strftime(...) was evaluated when config.py was
imported, which worked but was implicit and tied to import timing. Generating it
inside run_experiment() makes it explicit that the ID represents when the experiment
actually started, not when the config module happened to load.

## Type hints added
- run_experiment(...) -> dict[str, object]: clearly signals it returns a structured
  result dictionary
- log_result(result: dict[str, object], log_path: str | None = None) -> None: makes
  the expected input types explicit, improving readability and enabling tooling like
  Ruff/mypy to catch mistakes

## Behavior unchanged
Same config values produce identical metrics (verified: window_size=448 run after
refactor matched the pre-refactor 448 run exactly: rf_acc=0.4035, rf_bal_acc=0.565,
mlp_acc=0.2843, mlp_bal_acc=0.421). Refactoring improved structure without altering
the experiment pipeline itself.

## What this enables
Cleaner structure makes future extensions easier: adding new models, changing feature
extraction, or modifying the result format can now be done in focused functions
without touching unrelated code. This is the foundation for more advanced
experimentation workflows (hyperparameter sweeps, ablation studies, etc.).