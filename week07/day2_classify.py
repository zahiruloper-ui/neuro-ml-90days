import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

# --- Feature extraction: mean and std per channel ---
mean_features = X_windows.mean(axis=1)
std_features = X_windows.std(axis=1)
X_features = np.concatenate([mean_features, std_features], axis=1)

print("X_features shape:", X_features.shape)
print("y_windows shape:", y_windows.shape)

# --- Naive random train/test split (leakage risk — we'll fix this Day 3) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_windows, test_size=0.2, random_state=42
)

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #   fit makes it learn
X_test_scaled = scaler.transform(X_test)

# --- Train logistic regression ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train accuracy:", round(train_acc, 4))
print("Test accuracy:", round(test_acc, 4))

from sklearn.metrics import confusion_matrix

y_pred = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix (rows=true, cols=predicted):")
print(cm)
print("\nTrue negatives (correct eye-open):", cm[0, 0])
print("False positives (predicted closed, actually open):", cm[0, 1])
print("False negatives (predicted open, actually closed):", cm[1, 0])
print("True positives (correct eye-closed):", cm[1, 1])

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # no scaling needed for tree-based models

rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

rf_y_pred = rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_y_pred)

print("Random Forest train accuracy:", round(rf_train_acc, 4))
print("Random Forest test accuracy:", round(rf_test_acc, 4))
print("Random Forest confusion matrix:")
print(rf_cm)