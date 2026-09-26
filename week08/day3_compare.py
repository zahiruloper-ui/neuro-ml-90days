"""
Week 8 - Day 3 (Compare): Blocked time-series CV for RF vs tuned MLP.

Same EEG windowed features as Days 1–2. Instead of a single tail split,
we use K contiguous test blocks with a buffer around each block so
evaluation is more stable but still leakage-safe.

Run from repo root:
    .venv\Scripts\python.exe week8\day3\day3_blocked_compare.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# --- Config -----------------------------------------------------------------
WINDOW_SIZE = 128
STEP = 64
N_BLOCKS = 5     # number of contiguous test blocks
BUFFER = 1
SEED = 42

# Tuned MLP config from Day 2
MLP_HIDDEN = (64, 32)
MLP_ACTIVATION = "tanh"
MLP_MAX_ITER = 500

# RF config from Week 7 (adjust to your actual best params)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

# --- 1. Load ---------------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv")

label_col = "eyeDetection" if "eyeDetection" in df.columns else df.columns[-1]
X_raw = df.drop(columns=[label_col]).values.astype(float)
y_raw = df[label_col].values.astype(int)

print("Shape:", df.shape)
print("Label column:", label_col)
print("Label distribution:", df[label_col].value_counts().sort_index(), sep="\n")

# --- 2. Sliding windows ----------------------------------------------------
def create_sliding_windows(X, y, window_size=WINDOW_SIZE, step=STEP):
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        Xw.append(X[start:end])
        yw.append(int(y[start:end].mean() > 0.5))
    return np.array(Xw), np.array(yw)


# --- 3. Features -----------------------------------------------------------
def extract_features(Xw):
    return np.hstack([Xw.mean(axis=1), Xw.std(axis=1)])


Xw, y_windows = create_sliding_windows(X_raw, y_raw)
X_features = extract_features(Xw)

n_windows = len(X_features)
print("\nTotal windows:", n_windows)
print("Window label counts:", np.bincount(y_windows, minlength=2))

# --- 4. Build contiguous blocks -------------------------------------------
def make_block_indices(n_windows, n_blocks):
    # roughly equal-sized contiguous blocks
    indices = np.arange(n_windows)
    return np.array_split(indices, n_blocks)


block_indices = make_block_indices(n_windows, N_BLOCKS)

print("\nBlocked CV layout:")
for i, b in enumerate(block_indices):
    print(f"  Block {i}: windows {b[0]}–{b[-1]} (size {len(b)}) "
          f"labels={np.bincount(y_windows[b], minlength=2)}")

# --- 5. Build train/test splits with buffer -------------------------------
def blocked_time_splits(n_windows, block_indices, buffer=BUFFER):
    splits = []
    all_idx = np.arange(n_windows)
    for b in block_indices:
        min_t, max_t = b[0], b[-1]
        test_idx = b
        mask = (all_idx < min_t - buffer) | (all_idx > max_t + buffer)
        train_idx = all_idx[mask]
        splits.append((train_idx, test_idx))
    return splits


splits = blocked_time_splits(n_windows, block_indices, buffer=BUFFER)

print("\nBlocked splits (train/test sizes):")
for i, (train_idx, test_idx) in enumerate(splits):
    print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)}")

# --- 6. Models ------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_RANDOM_STATE,
)
mlp = MLPClassifier(
    hidden_layer_sizes=MLP_HIDDEN,
    activation=MLP_ACTIVATION,
    solver="adam",
    max_iter=MLP_MAX_ITER,
    random_state=SEED,
)

# --- 7. Run blocked CV ----------------------------------------------------
rf_fold_metrics = []
mlp_fold_metrics = []

print("\n--- Blocked CV metrics per fold ---")

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, y_train = X_features[train_idx], y_windows[train_idx]
    X_test, y_test = X_features[test_idx], y_windows[test_idx]

    # scale for MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # fit models
    rf.fit(X_train, y_train)
    mlp.fit(X_train_s, y_train)

    # RF metrics
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_bal = balanced_accuracy_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)

    # MLP metrics
    mlp_pred = mlp.predict(X_test_s)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_bal = balanced_accuracy_score(y_test, mlp_pred)
    mlp_cm = confusion_matrix(y_test, mlp_pred)

    rf_fold_metrics.append((rf_acc, rf_bal))
    mlp_fold_metrics.append((mlp_acc, mlp_bal))

    print(f"\nFold {fold}:")
    print(f"  RF  - acc={rf_acc:.3f}, bal_acc={rf_bal:.3f}")
    print("    RF confusion:\n", rf_cm)
    print(f"  MLP - acc={mlp_acc:.3f}, bal_acc={mlp_bal:.3f}")
    print("    MLP confusion:\n", mlp_cm)

# --- 8. Average metrics ----------------------------------------------------
rf_accs, rf_bals = zip(*rf_fold_metrics)
mlp_accs, mlp_bals = zip(*mlp_fold_metrics)

print("\n--- Average over folds ---")
print(f"RF  - mean acc={np.mean(rf_accs):.3f}, mean bal_acc={np.mean(rf_bals):.3f}")
print(f"MLP - mean acc={np.mean(mlp_accs):.3f}, mean bal_acc={np.mean(mlp_bals):.3f}")