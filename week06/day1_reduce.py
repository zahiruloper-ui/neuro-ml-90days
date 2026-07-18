# week6_day1_standardize_pca.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Task 1: Load EEG Eye State dataset (neuroscience/brain data)
print("Loading EEG Eye State dataset from OpenML...")
dataset = fetch_openml(name="eeg-eye-state", version=1)
df = dataset.data

print(f"Full dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Select only numerical columns (first 14 are EEG channels, last is eye state)
# Keep EEG features only (columns 0-13), drop eye state label (column 14)
eeg_cols = df.columns[:14]  # First 14 columns = EEG channels
X = df[eeg_cols]

print(f"\nEEG features only shape: {X.shape}")
print(f"EEG features: {list(X.columns)}")

# Task 1b: Standardize features
print("\nStandardizing features (mean=0, std=1)...")
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Verify standardization
means = np.mean(X_standardized, axis=0)
stds = np.std(X_standardized, axis=0)
print(f"Mean after standardization (should be ~0): {means[:3]}...")
print(f"Std after standardization (should be ~1): {stds[:3]}...")

print("\nTask 1 complete: EEG dataset loaded and standardized.")

# week6_day1_standardize_pca.py (continuation from Task 1)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Task 2: Run PCA on standardized EEG data
print("\nRunning PCA on standardized EEG data...")
pca = PCA(n_components=None, random_state=42)
X_pca = pca.fit_transform(X_standardized)

print(f"PCA output shape: {X_pca.shape}")
print(f"Number of components: {len(pca.explained_variance_ratio_)}")

# Inspect explained variance
print("\nExplained variance ratio per PC:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")

# Cumulative variance
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative variance explained:")
for i, cum_var in enumerate(cumulative_var, 1):
    print(f"  Top {i} PCs: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Find how many PCs to capture 95% of variance
n_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nPCs needed to capture 95% of variance: {n_95}")

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'bo-', label='Individual')
plt.plot(range(1, len(cumulative_var) + 1), 
         cumulative_var, 'rs-', label='Cumulative')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance per Component (EEG Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('week6_day1_pca_variance.png', dpi=150)
print("\nPlot saved: week6_day1_pca_variance.png")

# week6_day1_standardize_pca.py (continuation from Task 2)

# Task 3: Refit PCA with 5 components and interpret
print("\n--- Task 3: Refitting PCA with 5 components ---")
pca_5 = PCA(n_components=5, random_state=42)
X_reduced = pca_5.fit_transform(X_standardized)

print(f"Reduced data shape: {X_reduced.shape}")
print(f"Explained variance (5 PCs): {pca_5.explained_variance_ratio_.sum():.4f} ({pca_5.explained_variance_ratio_.sum()*100:.2f}%)")

# Inspect feature loadings (which EEG channels matter most?)
print("\nTop contributions to each PC (absolute loadings):")
feature_names = list(X.columns)

for pc in range(5):
    loadings = pca_5.components_[pc]
    sorted_idx = np.argsort(np.abs(loadings))[::-1]  # Descending by absolute value
    print(f"\nPC{pc+1} (explains {pca_5.explained_variance_ratio_[pc]*100:.1f}%):")
    for i in sorted_idx[:3]:  # Top 3 features per PC
        print(f"  {feature_names[i]}: {loadings[i]:+.3f}")

# Save reduced data for Day 2 clustering
import joblib
joblib.dump(X_reduced, 'week6_X_reduced_5pc.joblib')
joblib.dump(pca_5, 'week6_pca_model.joblib')
print("\nSaved: week6_X_reduced_5pc.joblib (reduced data for clustering)")
print("Saved: week6_pca_model.joblib (PCA model)")