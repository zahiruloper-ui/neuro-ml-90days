import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"

WINDOW_SIZE = 128
STEP = 64
TRAIN_FRAC = 0.8
BUFFER = 1
SEED = 42


def load_eeg_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y


def create_sliding_windows(X, y, window_size=WINDOW_SIZE, step=STEP):
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        Xw.append(X[start:end])
        vals, counts = np.unique(y[start:end], return_counts=True) 

        yw.append(vals[np.argmax(counts)])   # assigning one label to each window
    return np.array(Xw), np.array(yw)


def extract_features(Xw):
    means = Xw.mean(axis=1)
    stds = Xw.std(axis=1)
    return np.hstack([means, stds])


def time_block_split(n_windows, train_frac=TRAIN_FRAC, buffer=BUFFER):
    split_idx = int(n_windows * train_frac)
    train_idx = np.arange(0, split_idx - buffer)
    test_idx = np.arange(split_idx + buffer, n_windows)
    return train_idx, test_idx


X_raw, y_raw = load_eeg_data(DATA_PATH)
Xw, yw = create_sliding_windows(X_raw, y_raw)
X_features = extract_features(Xw)

train_idx, test_idx = time_block_split(len(X_features))
X_train, y_train = X_features[train_idx], yw[train_idx]
X_test, y_test = X_features[test_idx], yw[test_idx]

print("X_features shape:", X_features.shape)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train class counts:", np.bincount(y_train))
print("Test class counts:", np.bincount(y_test))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=SEED,
)
mlp.fit(X_train_s, y_train)

train_acc = accuracy_score(y_train, mlp.predict(X_train_s))
test_acc = accuracy_score(y_test, mlp.predict(X_test_s))

print("\n--- Baseline MLP (32 hidden units, relu) ---")
print("Iterations run:", mlp.n_iter_)
print("Final training loss:", round(mlp.loss_, 4))
print("MLP train accuracy:", round(train_acc, 3))
print("MLP test accuracy:", round(test_acc, 3))
print("MLP confusion matrix (test):")
print(confusion_matrix(y_test, mlp.predict(X_test_s)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

DATA_URL = "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
WINDOW_SIZE, STEP, TRAIN_FRAC, BUFFER, SEED = 128, 64, 0.8, 1, 42


def load_eeg_data(url=DATA_URL):
    df = pd.read_csv(url)
    return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(int)


def create_sliding_windows(X, y, window_size=WINDOW_SIZE, step=STEP):
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, step):
        Xw.append(X[start:start + window_size])
        yw.append(int(y[start:start + window_size].mean() > 0.5))
    return np.array(Xw), np.array(yw)


def extract_features(Xw):
    return np.hstack([Xw.mean(axis=1), Xw.std(axis=1)])


def time_block_split(n, train_frac=TRAIN_FRAC, buffer=BUFFER):
    s = int(n * train_frac)
    return np.arange(0, s - buffer), np.arange(s + buffer, n)


X_raw, y_raw = load_eeg_data()
Xw, yw = create_sliding_windows(X_raw, y_raw)
F = extract_features(Xw)
tr, te = time_block_split(len(F))
X_train, y_train, X_test, y_test = F[tr], yw[tr], F[te], yw[te]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("--- 1. Distribution shift ---")
print("Train class counts:", np.bincount(y_train, minlength=2))
print("Test  class counts:", np.bincount(y_test, minlength=2))
print("Max |z| scaled train:", round(np.abs(X_train_s).max(), 1))
print("Max |z| scaled test :", round(np.abs(X_test_s).max(), 1))

print("\n--- 2. Dummy baselines on test ---")
for strat in ["most_frequent", "stratified", "uniform"]:
    d = DummyClassifier(strategy=strat, random_state=SEED).fit(X_train, y_train)
    print(f"{strat:>14}: acc={accuracy_score(y_test, d.predict(X_test)):.3f}")

mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=SEED)
mlp.fit(X_train_s, y_train)
rf = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

print("\n--- 3. Loss curve (every 50 epochs) ---")
lc = mlp.loss_curve_
print([round(lc[i], 3) for i in range(0, len(lc), 50)])
print("Last epoch loss:", round(lc[-1], 4),
      "| still decreasing?", lc[-1] < lc[-50])

print("\n--- 4. Metrics that survive imbalance ---")
for name, model, Xt in [("MLP", mlp, X_test_s), ("RF", rf, X_test)]:
    p = model.predict(Xt)
    print(f"\n{name}: pred counts={np.bincount(p, minlength=2)}")
    print(f"  accuracy          = {accuracy_score(y_test, p):.3f}")
    print(f"  balanced accuracy = {balanced_accuracy_score(y_test, p):.3f}")
    print("  confusion matrix:\n", confusion_matrix(y_test, p))
    print(classification_report(y_test, p, zero_division=0))