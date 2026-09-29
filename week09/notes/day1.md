# Week 9, Day 1 — Pin

## What "reproducible" means
Same code + same data + same config + same package versions -> same results, every time,
on any machine. If any one of those four things changes, results can silently change too.

## Why pinning matters
Installing a package without a version number (e.g. `pip install pandas`) gets whatever
is newest *right now*. Six months later, the newest version might behave differently,
change a default value, or even remove a function. Pinning means writing the exact
version used (`pandas==3.0.1`), so anyone reinstalling later gets the same package,
not just the same package name.

## What pinning does NOT protect against
Pinning packages only locks down *which code runs*. It does not lock down:
- **Data** — if the dataset file changes or gets updated upstream, results change
  even with identical code and packages.
- **Random seeds** — without an explicit seed, models get different random numbers
  each run (train/test splits, weight initialization), causing different results
  even on the same machine with the same packages.
- **Hardware/architecture** — different CPUs (e.g. ARM vs Intel/AMD) can very rarely
  cause tiny floating-point differences in low-level numerical operations.
- **Operating system** — occasionally causes subtle library behavior differences.

Pinning is necessary but not sufficient for full reproducibility.

## What we built today
- `requirements.txt` — exact versions of all 18 packages in the `.venv`, generated
  with `pip freeze`. Confirmed core packages pinned: numpy, pandas, scikit-learn.
- `environment_info.txt` — Python version, OS platform, machine architecture, processor,
  and pip version, generated with a small Python script using `sys` and `platform`.

## Lightweight tracking vs. full MLOps tools (MLflow, Weights & Biases)
Full tools add things a simple file/CSV setup doesn't automate:
- **Run comparison dashboards** — browsing/sorting/plotting many runs visually instead
  of manually reading rows in a spreadsheet. Useful at hundreds of runs or with a team;
  overkill for a handful of solo experiments.
- **Automated artifact storage/versioning** — automatically saving and linking model
  files, plots, etc. to a specific run ID. A solo project can do this manually (save
  file, write its path in a log row) at much lower cost.

The lightweight approach (pinned deps + config + CSV log) matches the actual scale of
a solo student project. Heavier tools become worth the setup cost at team scale or when
running very large numbers of experiments.

## Note on this machine's setup
This project's Python (3.11.9) is running via x64 emulation on ARM64 (Qualcomm Snapdragon)
hardware, not natively. This is recorded in `environment_info.txt` (Machine: AMD64,
Processor: ARMv8 ... Qualcomm) so any odd behavior later has a documented possible cause.