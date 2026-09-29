import numpy as np
from eeg_pipeline import (
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

# --- 1. Configs ------------------------------------------------------------

data_cfg = DatasetConfig(csv_path="https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv")
split_cfg = SplitConfig(n_blocks=5, buffer=1)
model_cfg = ModelConfig(
    rf_n_estimators=200,
    rf_max_depth=None,
    rf_random_state=42,
    mlp_hidden=(64, 32),
    mlp_activation="tanh",
    mlp_max_iter=500,
    mlp_random_state=42,
)

# --- 2. Load + pipeline ----------------------------------------------------

X_raw, y_raw = load_eeg_dataset(data_cfg)
Xw, yw = create_sliding_windows(X_raw, y_raw, data_cfg.window_size, data_cfg.step)
X_features = extract_features(Xw)

print("X_raw shape:", X_raw.shape)
print("Total windows:", len(X_features))
print("Window label counts:", np.bincount(yw, minlength=2))

blocks = make_block_indices(len(X_features), split_cfg.n_blocks)
print("\nBlocked layout:")
for i, b in enumerate(blocks):
    print(
        f"  Block {i}: win {b[0]}-{b[-1]} size={len(b)} labels="
        f"{np.bincount(yw[b], minlength=2)}"
    )

splits = blocked_time_splits(len(X_features), blocks, split_cfg.buffer)
print("\nBlocked splits (train/test sizes):")
for i, (train_idx, test_idx) in enumerate(splits):
    print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)})")

# --- 3. Run blocked CV via common module ----------------------------------

results = run_blocked_cv(X_features, yw, splits, model_cfg)

print("\n--- Per-fold metrics (via refactored pipeline) ---")
for r in results:
    print(
        f"Fold {r.fold}: RF acc={r.rf_acc:.3f}, bal_acc={r.rf_bal_acc:.3f}; "
        f"MLP acc={r.mlp_acc:.3f}, bal_acc={r.mlp_bal_acc:.3f}"
    )

rf_mean_acc, rf_mean_bal, mlp_mean_acc, mlp_mean_bal = summarize_results(results)

print("\n--- Mean metrics ---")
print(f"RF  mean acc={rf_mean_acc:.3f}, mean bal_acc={rf_mean_bal:.3f}")
print(f"MLP mean acc={mlp_mean_acc:.3f}, mean bal_acc={mlp_mean_bal:.3f}")
