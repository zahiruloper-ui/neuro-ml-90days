import numpy as np
import pandas as pd

# --- Load data ---
df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
)
eeg_cols = [c for c in df.columns if c != "class"]
X = df[eeg_cols].to_numpy()
y = df["class"].to_numpy()

window_size = 128
step = 64
n_samples, n_channels = X.shape

# Recreate window start/end indices (not the data itself, just the index ranges)
window_indices = []
for start in range(0, n_samples - window_size + 1, step):
    end = start + window_size
    window_indices.append((start, end))

print("Total windows:", len(window_indices))

# Inspect two consecutive windows: window 0 and window 1
start_a, end_a = window_indices[0]
start_b, end_b = window_indices[1]

print(f"Window 0 covers samples [{start_a}, {end_a})")
print(f"Window 1 covers samples [{start_b}, {end_b})")

# Compute overlap
overlap_start = max(start_a, start_b)
overlap_end = min(end_a, end_b)
overlap_count = max(0, overlap_end - overlap_start)

print("Overlapping samples between window 0 and window 1:", overlap_count)
print("Fraction of window size that is overlap:", round(overlap_count / window_size, 3))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# --- Rebuild windows and features (same as Day 1/2) ---
windows = []
window_labels = []
for start in range(0, n_samples - window_size + 1, step):
    end = start + window_size
    windows.append(X[start:end, :])
    window_labels.append(y[end - 1])

X_windows = np.stack(windows)
y_windows = np.array(window_labels)

mean_features = X_windows.mean(axis=1)
std_features = X_windows.std(axis=1)
X_features = np.concatenate([mean_features, std_features], axis=1)

n_windows = X_features.shape[0]

# --- Time-block split with buffer to avoid overlap leakage ---
split_idx = int(n_windows * 0.8)
buffer = 1  # drop 1 window on each side of the boundary

train_idx = np.arange(0, split_idx - buffer)
test_idx = np.arange(split_idx + buffer, n_windows)

X_train_tb, y_train_tb = X_features[train_idx], y_windows[train_idx]
X_test_tb, y_test_tb = X_features[test_idx], y_windows[test_idx]

print("Time-block train shape:", X_train_tb.shape)
print("Time-block test shape:", X_test_tb.shape)

# --- Logistic Regression on time-block split ---
scaler = StandardScaler()
X_train_tb_scaled = scaler.fit_transform(X_train_tb)
X_test_tb_scaled = scaler.transform(X_test_tb)

logreg_tb = LogisticRegression(max_iter=1000)
logreg_tb.fit(X_train_tb_scaled, y_train_tb)
logreg_tb_train_acc = logreg_tb.score(X_train_tb_scaled, y_train_tb)
logreg_tb_test_acc = logreg_tb.score(X_test_tb_scaled, y_test_tb)

print("\n[Time-block] Logistic Regression train accuracy:", round(logreg_tb_train_acc, 4))
print("[Time-block] Logistic Regression test accuracy:", round(logreg_tb_test_acc, 4))

# --- Random Forest on time-block split ---
rf_tb = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tb.fit(X_train_tb, y_train_tb)
rf_tb_train_acc = rf_tb.score(X_train_tb, y_train_tb)
rf_tb_test_acc = rf_tb.score(X_test_tb, y_test_tb)
rf_tb_cm = confusion_matrix(y_test_tb, rf_tb.predict(X_test_tb))

print("\n[Time-block] Random Forest train accuracy:", round(rf_tb_train_acc, 4))
print("[Time-block] Random Forest test accuracy:", round(rf_tb_test_acc, 4))
print("[Time-block] Random Forest confusion matrix:")
print(rf_tb_cm)
