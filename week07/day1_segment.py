import numpy as np
import pandas as pd

# Load the standard EEG Eye State CSV
df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
)

eeg_cols = [c for c in df.columns if c != "class"]
X = df[eeg_cols].to_numpy()        # shape: (n_samples, n_channels)
y = df["class"].to_numpy()         # shape: (n_samples,)

# --- Sliding window parameters ---
window_size = 128   # number of time samples per window
step = 64           # step between windows (50% overlap)

n_samples, n_channels = X.shape

windows = []
window_labels = []

# Create sliding windows
for start in range(0, n_samples - window_size + 1, step):
    end = start + window_size
    window = X[start:end, :]          # shape: (window_size, n_channels)
    label = y[end - 1]                # label at the last time point in the window # it takes from y (which has the class column)
    windows.append(window)
    window_labels.append(label)

X_windows = np.stack(windows)         # shape: (n_windows, window_size, n_channels) np.stack enforces the assumption 
                                      # “all windows are the same size” and 
                                      # makes the new “window index” dimension explicit.
y_windows = np.array(window_labels)   # shape: (n_windows,)

print("Window size (samples):", window_size)
print("Step size (samples):", step)
print("Number of windows created:", X_windows.shape[0])  # here 0 is index for X_windows(n_windows, window_size, n_channels)
print("X_windows shape:", X_windows.shape)  # (n_windows, window_size, n_channels)
print("y_windows shape:", y_windows.shape)
print("Label distribution in windows:\n", np.bincount(y_windows)) # counts how many times
                                                                  # each class label appears in y_window
# --- Feature extraction: mean and std per channel, per window ---
mean_features = X_windows.mean(axis=1)   # shape: (n_windows, n_channels)
std_features = X_windows.std(axis=1)     # shape: (n_windows, n_channels)

X_features = np.concatenate([mean_features, std_features], axis=1)  # shape: (n_windows, n_channels*2)

print("mean_features shape:", mean_features.shape)
print("std_features shape:", std_features.shape)
print("X_features shape:", X_features.shape)
print("\nFirst window's feature vector (first 6 values):\n", X_features[0][:6])