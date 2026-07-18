# week6_day2_kmeans.py
import numpy as np
import joblib

# Task 1: Load reduced data from Day 1
print("Loading reduced EEG data (5 PCs)...")
X_reduced = joblib.load('week6_X_reduced_5pc.joblib')
print(f"Data shape: {X_reduced.shape}")

# Task 1b: Run k-means with k=3
print("\nRunning k-means clustering (k=3)...")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_reduced)

print(f"Cluster assignments shape: {cluster_labels.shape}")
print(f"Number of samples per cluster:")
for k in range(3):
    count = np.sum(cluster_labels == k)
    print(f"  Cluster {k}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

print(f"\nInertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Save cluster labels for Day 3 visualization
joblib.dump(cluster_labels, 'week6_cluster_labels_k3.joblib')
print("\nSaved: week6_cluster_labels_k3.joblib")


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Task 2: Test multiple k values (elbow method)
print("\n--- Task 2: Testing k from 2 to 10 ---")
k_range = range(2, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_reduced)
    inertias.append(kmeans.inertia_)
    print(f"k={k}: inertia={kmeans.inertia_:.2f}")

# Find best k using elbow (visual inspection)
# Calculate "elbow score" as second derivative (rate of change of rate of change)
if len(inertias) >= 3:
    # First derivative (rate of decrease)
    first_diff = np.diff(inertias)
    # Second derivative (change in rate)
    second_diff = np.diff(first_diff)
    # Elbow is where second derivative is maximum (curve bends most)
    elbow_k = k_range[np.argmax(second_diff) + 2]  # +2 because diff reduces length by 2
    print(f"\nElbow method suggests k = {elbow_k}")
else:
    elbow_k = 2
    print("\nElbow method inconclusive, using k=2")

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.title('Elbow Method: Choosing k for EEG Clustering')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)

# Mark elbow point
plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
plt.legend()
plt.tight_layout()
plt.savefig('week6_day2_elbow_plot.png', dpi=150)
print(f"\nPlot saved: week6_day2_elbow_plot.png")

# Re-run k-means with best k
print(f"\nRe-running k-means with optimal k={elbow_k}...")
kmeans_best = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
cluster_labels_best = kmeans_best.fit_predict(X_reduced)

print(f"Cluster assignments shape: {cluster_labels_best.shape}")
print(f"Number of samples per cluster:")
for k in range(elbow_k):
    count = np.sum(cluster_labels_best == k)
    print(f"  Cluster {k}: {count} samples ({count/len(cluster_labels_best)*100:.1f}%)")

# Save best clustering
joblib.dump(cluster_labels_best, 'week6_cluster_labels_best.joblib')
joblib.dump(kmeans_best, 'week6_kmeans_model_best.joblib')
print(f"\nSaved: week6_cluster_labels_best.joblib")
print(f"Saved: week6_kmeans_model_best.joblib")

# week6_day2_kmeans.py (continuation)

# Task 3: Try k=2 (simplest split)
print("\n--- Task 3: Testing k=2 (simplest split) ---")
kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = kmeans_k2.fit_predict(X_reduced)

print(f"Cluster distribution for k=2:")
for k in range(2):
    count = np.sum(labels_k2 == k)
    print(f"  Cluster {k}: {count} samples ({count/len(labels_k2)*100:.1f}%)")

print(f"Inertia: {kmeans_k2.inertia_:.2f}")

# Save k=2 results
joblib.dump(labels_k2, 'week6_cluster_labels_k2.joblib')
print("\nSaved: week6_cluster_labels_k2.joblib")

