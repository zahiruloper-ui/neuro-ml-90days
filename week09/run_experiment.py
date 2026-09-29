import sys
from pathlib import Path

WEEK8_DIR = Path(__file__).resolve().parent.parent / "week08"
sys.path.insert(0, str(WEEK8_DIR))

from eeg_pipeline import (
    DatasetConfig,
    SplitConfig,
    ModelConfig,
    load_eeg_dataset,
    create_sliding_windows,
    extract_features,
    make_block_indices,
    blocked_time_splits,
    run_blocked_cv,
    summarize_results,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day2_config import (
    SEED,
    DATA_PATH,
    WINDOW_SIZE,
    STEP,
    N_BLOCKS,
    BUFFER,
    RF_PARAMS,
    MLP_PARAMS,
    RUN_ID,
)


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
    print(f"=== Run {result['run_id']} ===")
    for key, value in result.items():
        print(f"{key}: {value}")