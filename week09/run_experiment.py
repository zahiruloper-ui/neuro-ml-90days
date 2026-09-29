import csv
import os
import sys
from pathlib import Path

import pandas as pd

WEEK8_DIR = Path(__file__).resolve().parent.parent / "week08"
sys.path.insert(0, str(WEEK8_DIR))

from eeg_pipeline import (  # type: ignore
    DatasetConfig,
    ModelConfig,
    SplitConfig,
    blocked_time_splits,
    create_sliding_windows,
    extract_features,
    load_eeg_dataset,
    make_block_indices,
    run_blocked_cv,
    summarize_results,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BUFFER,
    DATA_PATH,
    MLP_PARAMS,
    N_BLOCKS,
    RF_PARAMS,
    RUN_ID,
    SEED,
    STEP,
    WINDOW_SIZE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def log_result(result, log_path=None):
    if log_path is None:
        log_path = os.path.join(SCRIPT_DIR, "results_log.csv")
    file_exists = os.path.isfile(log_path)
    row = {
        "run_id": result["run_id"],
        "seed": result["seed"],
        "window_size": result["window_size"],
        "step": result["step"],
        "n_blocks": result["n_blocks"],
        "buffer": result["buffer"],
        "rf_params": str(result["rf_params"]),
        "mlp_params": str(result["mlp_params"]),
        "n_windows": result["n_windows"],
        "rf_acc": result["rf_acc"],
        "rf_bal_acc": result["rf_bal_acc"],
        "mlp_acc": result["mlp_acc"],
        "mlp_bal_acc": result["mlp_bal_acc"],
    }
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> dict:
    data_cfg = DatasetConfig(
        csv_path=DATA_PATH,
        window_size=WINDOW_SIZE,
        step=STEP,
    )
    split_cfg = SplitConfig(n_blocks=N_BLOCKS, buffer=BUFFER)
    model_cfg = ModelConfig(
        rf_n_estimators=RF_PARAMS["n_estimators"],
        rf_max_depth=RF_PARAMS["max_depth"],
        rf_random_state=RF_PARAMS["random_state"],
        mlp_hidden=MLP_PARAMS["hidden_layer_sizes"],
        mlp_activation=MLP_PARAMS["activation"],
        mlp_max_iter=MLP_PARAMS["max_iter"],
        mlp_random_state=MLP_PARAMS["random_state"],
    )

    X_raw, y_raw = load_eeg_dataset(data_cfg)
    Xw, yw = create_sliding_windows(X_raw, y_raw, data_cfg.window_size, data_cfg.step)
    X_features = extract_features(Xw)

    block_idx = make_block_indices(len(Xw), split_cfg.n_blocks)
    splits = blocked_time_splits(len(Xw), block_idx, split_cfg.buffer)

    results = run_blocked_cv(X_features, yw, splits, model_cfg)
    rf_acc, rf_bal, mlp_acc, mlp_bal = summarize_results(results)

    return {
        "run_id": RUN_ID,
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "n_blocks": N_BLOCKS,
        "buffer": BUFFER,
        "rf_params": RF_PARAMS,
        "mlp_params": MLP_PARAMS,
        "n_windows": len(Xw),
        "rf_acc": round(rf_acc, 4),
        "rf_bal_acc": round(rf_bal, 4),
        "mlp_acc": round(mlp_acc, 4),
        "mlp_bal_acc": round(mlp_bal, 4),
    }


if __name__ == "__main__":
    result = main()
    log_result(result)
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "results_log.csv"))
    print(df.to_string(index=False))
    for key, value in result.items():
        print(f"{key}: {value}")
