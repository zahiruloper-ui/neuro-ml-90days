import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# --- Config -----------------------------------------------------------------
WINDOW_SIZE = 128
STEP = 64
N_BLOCKS = 5
BUFFER = 1
SEED = 42

# Tuned MLP (from Day 2/3)
MLP_HIDDEN = (64, 32)
MLP_ACTIVATION = "tanh"
MLP_MAX_ITER = 500

# RF config (Week 7 best)
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
print("Label distribution:\n", df[label_col].value_counts().sort_index())

# --- 2. Sliding windows + features ----------------------------------------

def create_sliding_windows(X, y, window_size=WINDOW_SIZE, step=STEP):
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        Xw.append(X[start:end])
        yw.append(int(y[start:end].mean() > 0.5))
    return np.array(Xw), np.array(yw)


def extract_features(Xw):
    return np.hstack([Xw.mean(axis=1), Xw.std(axis=1)])


Xw, y_windows = create_sliding_windows(X_raw, y_raw)
X_features = extract_features(Xw)

n_windows = len(X_features)
print("\nTotal windows:", n_windows)
print("Window label counts:", np.bincount(y_windows, minlength=2))

# --- 3. Blocked splits (reuse Day 3 logic) ---------------------------------

def make_block_indices(n_windows, n_blocks):
    indices = np.arange(n_windows)
    return np.array_split(indices, n_blocks)


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


block_indices = make_block_indices(n_windows, N_BLOCKS)

print("\nBlocked CV layout:")
for i, b in enumerate(block_indices):
    print(
        f"  Block {i}: win {b[0]}-{b[-1]} size={len(b)} labels="
        f"{np.bincount(y_windows[b], minlength=2)}"
    )

splits = blocked_time_splits(n_windows, block_indices, buffer=BUFFER)

print("\nBlocked splits (train/test sizes):")
for i, (train_idx, test_idx) in enumerate(splits):
    print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)})")

# --- 4. Models -------------------------------------------------------------
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

# --- 5. Agreement / disagreement analysis ---------------------------------

# Global counters across all folds
both_correct = 0
both_wrong = 0
rf_only_correct = 0
mlp_only_correct = 0

# Per-fold summaries
fold_summaries = []

print("\n--- Per-fold agreement/disagreement ---")

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, y_train = X_features[train_idx], y_windows[train_idx]
    X_test, y_test = X_features[test_idx], y_windows[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf.fit(X_train, y_train)
    mlp.fit(X_train_s, y_train)

    rf_pred = rf.predict(X_test)
    mlp_pred = mlp.predict(X_test_s)

    rf_acc = accuracy_score(y_test, rf_pred)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    rf_bal = balanced_accuracy_score(y_test, rf_pred)
    mlp_bal = balanced_accuracy_score(y_test, mlp_pred)

    # per-fold counts
    bc = bw = roc = moc = 0

    for i, idx in enumerate(test_idx):
        true = y_test[i]
        r = rf_pred[i]
        m = mlp_pred[i]
        if r == true and m == true:
            bc += 1
            both_correct += 1
        elif r != true and m != true:
            bw += 1
            both_wrong += 1
        elif r == true and m != true:
            roc += 1
            rf_only_correct += 1
        elif r != true and m == true:
            moc += 1
            mlp_only_correct += 1

    fold_summaries.append((bc, bw, roc, moc, rf_acc, mlp_acc, rf_bal, mlp_bal))

    print(f"\nFold {fold}:")
    print(f"  RF  - acc={rf_acc:.3f}, bal_acc={rf_bal:.3f}")
    print(f"  MLP - acc={mlp_acc:.3f}, bal_acc={mlp_bal:.3f}")
    print(f"  both_correct     = {bc}")
    print(f"  both_wrong       = {bw}")
    print(f"  RF_only_correct  = {roc}")
    print(f"  MLP_only_correct = {moc}")

# --- 6. Global summary -----------------------------------------------------

print("\n--- Global agreement/disagreement across all folds ---")
print("both_correct     =", both_correct)
print("both_wrong       =", both_wrong)
print("RF_only_correct  =", rf_only_correct)
print("MLP_only_correct =", mlp_only_correct)

total_test = both_correct + both_wrong + rf_only_correct + mlp_only_correct
print("total_test_windows =", total_test)

print("\nFractions:")
print("  frac_both_correct    =", round(both_correct / total_test, 3))
print("  frac_both_wrong      =", round(both_wrong / total_test, 3))
print("  frac_RF_only_correct =", round(rf_only_correct / total_test, 3))
print("  frac_MLP_only_correct=", round(mlp_only_correct / total_test, 3))

# --- 7. Detailed rows for one fold (e.g., tail fold) ----------------------

TAIL_FOLD = 4
train_idx_tail, test_idx_tail = splits[TAIL_FOLD]
X_train_tail, y_train_tail = X_features[train_idx_tail], y_windows[train_idx_tail]
X_test_tail, y_test_tail = X_features[test_idx_tail], y_windows[test_idx_tail]

scaler = StandardScaler()
X_train_tail_s = scaler.fit_transform(X_train_tail)
X_test_tail_s = scaler.transform(X_test_tail)

rf.fit(X_train_tail, y_train_tail)
mlp.fit(X_train_tail_s, y_train_tail)

rf_pred_tail = rf.predict(X_test_tail)
mlp_pred_tail = mlp.predict(X_test_tail_s)

print("\n--- Detailed tail-fold rows (Fold 4) ---")
print(f"{'win_idx':>7} {'true':>5} {'rf':>5} {'mlp':>5} {'case':>10}")
for i, idx in enumerate(test_idx_tail):
    true = y_test_tail[i]
    r = rf_pred_tail[i]
    m = mlp_pred_tail[i]
    if r == true and m == true:
        case = "both_correct"
    elif r != true and m != true:
        case = "both_wrong"
    elif r == true and m != true:
        case = "RF_only_correct"
    else:
        case = "MLP_only_correct"
    print(f"{idx:7d} {true:5d} {r:5d} {m:5d} {case:>10}")