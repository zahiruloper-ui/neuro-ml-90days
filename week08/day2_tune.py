import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# --- Config -----------------------------------------------------------------
WINDOW_SIZE = 128
STEP = 64
TRAIN_FRAC = 0.8
BUFFER = 1
SEED = 42

# Hyperparameter grid for Day 2 Task 1
HIDDEN_CONFIGS = [
    (16,),
    (32,),
    (64,),
    (32, 32),
    (64, 32),
]
ACTIVATIONS = ["relu", "tanh"]
MAX_ITERS = [500, 2000]


df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

label_col = "eyeDetection" if "eyeDetection" in df.columns else df.columns[-1]
print("Label column:", label_col)
print("Label distribution:\n", df[label_col].value_counts().sort_index())

X_raw = df.drop(columns=[label_col]).values.astype(float)
y_raw = df[label_col].values.astype(int)


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


# --- 4. Leakage-safe time-block split -------------------------------------
def time_block_split(n_windows, train_frac=TRAIN_FRAC, buffer=BUFFER):
    split_idx = int(n_windows * train_frac)
    train_idx = np.arange(0, split_idx - buffer)
    test_idx = np.arange(split_idx + buffer, n_windows)
    return train_idx, test_idx


Xw, y_windows = create_sliding_windows(X_raw, y_raw)
X_features = extract_features(Xw)
train_idx, test_idx = time_block_split(len(X_features))

X_train, y_train = X_features[train_idx], y_windows[train_idx]
X_test, y_test = X_features[test_idx], y_windows[test_idx]

print("\nX_features shape:", X_features.shape)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train class counts:", np.bincount(y_train, minlength=2))
print("Test class counts:", np.bincount(y_test, minlength=2))

# --- 5. Scale features (fit on TRAIN only) ---------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# --- 6. Sweep MLP configs --------------------------------------------------
print("\n--- Day 2 Task 1: MLP hyperparameter sweep ---")
print("(Metrics on the SAME leakage-safe split as Day 1)\n")

header = (
    f"{'hidden':>12} {'act':>6} {'max_iter':>9} "
    f"{'train_acc':>10} {'test_acc':>10} {'bal_acc':>10}"
)
print(header)
print("-" * len(header))

results = []

for hidden in HIDDEN_CONFIGS:
    for act in ACTIVATIONS:
        for mi in MAX_ITERS:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=act,
                solver="adam",
                max_iter=mi,
                random_state=SEED,
            )
            mlp.fit(X_train_s, y_train)

            y_pred_train = mlp.predict(X_train_s)
            y_pred_test = mlp.predict(X_test_s)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred_test)

            line = (
                f"{str(hidden):>12} {act:>6} {mi:>9} "
                f"{train_acc:>10.3f} {test_acc:>10.3f} {bal_acc:>10.3f}"
            )
            print(line)

            results.append(
                (
                    hidden,
                    act,
                    mi,
                    train_acc,
                    test_acc,
                    bal_acc,
                    confusion_matrix(y_test, y_pred_test),
                )
            )

print("\n--- Confusion matrices for each config (test set) ---")
for hidden, act, mi, train_acc, test_acc, bal_acc, cm in results:
    print(
        f"\nConfig hidden={hidden}, act={act}, max_iter={mi}: "
        f"train={train_acc:.3f}, test={test_acc:.3f}, bal={bal_acc:.3f}"
    )
    print(cm)