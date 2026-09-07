import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Load data ---
df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
)
eeg_cols = [c for c in df.columns if c != "class"]
X = df[eeg_cols].to_numpy()
y = df["class"].to_numpy()

# --- Sliding windows ---
window_size = 128
step = 64
n_samples, n_channels = X.shape

windows = []
window_labels = []
for start in range(0, n_samples - window_size + 1, step):
    end = start + window_size
    windows.append(X[start:end, :])
    window_labels.append(y[end - 1])

X_windows = np.stack(windows)
y_windows = np.array(window_labels)

# --- Feature extraction ---
mean_features = X_windows.mean(axis=1)
std_features = X_windows.std(axis=1)
X_features = np.concatenate([mean_features, std_features], axis=1)

n_windows = X_features.shape[0]

# --- Time-block split (same as Day 3) ---
split_idx = int(n_windows * 0.8)
buffer = 1

train_idx = np.arange(0, split_idx - buffer)
test_idx = np.arange(split_idx + buffer, n_windows)

X_train_tb, y_train_tb = X_features[train_idx], y_windows[train_idx]
X_test_tb, y_test_tb = X_features[test_idx], y_windows[test_idx]

# --- Random Forest ---
rf_tb = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tb.fit(X_train_tb, y_train_tb)
y_pred_tb = rf_tb.predict(X_test_tb)

# --- Per-window error table ---
correct = (y_pred_tb == y_test_tb)

print(f"{'TestIdx':>8} {'WinIdx':>7} {'True':>5} {'Pred':>5} {'Correct':>8}")
for i, win_idx in enumerate(test_idx):
    print(f"{i:>8} {win_idx:>7} {y_test_tb[i]:>5} {y_pred_tb[i]:>5} {str(correct[i]):>8}")

print("\nTotal test windows:", len(test_idx))
print("Total errors:", (~correct).sum())
print("Error indices (position within test block):", np.where(~correct)[0].tolist())


rolling_acc = pd.Series(correct.astype(int)).rolling(window=5).mean()

print(f"{'TestIdx':>8} {'RollingAcc(5)':>14}")
for i in range(len(correct)):
    val = rolling_acc[i]
    val_str = f"{val:.2f}" if not pd.isna(val) else "NA"
    print(f"{i:>8} {val_str:>14}") 