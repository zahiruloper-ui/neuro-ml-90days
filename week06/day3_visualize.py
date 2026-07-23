# week6_day3_visualize.py
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Task 1: Load reduced data and cluster labels
print("Loading reduced EEG data (5 PCs) and cluster labels...")
X_reduced = joblib.load('week6_X_reduced_5pc.joblib')
labels_k2 = joblib.load('week6_cluster_labels_k2.joblib')

print(f"Data shape: {X_reduced.shape}")
print(f"Cluster labels shape: {labels_k2.shape}")

# Extract PC1 and PC2 (first 2 columns of 5-PC data)
pc1 = X_reduced[:, 0]
pc2 = X_reduced[:, 1]

# Get variance explained by PC1 and PC2 (from Day 1 output)
# PC1: 29.25%, PC2: 23.90%
var_pc1 = 29.25
var_pc2 = 23.90

# Task 1b: Create 2D scatter plot
print("\nCreating 2D PCA scatter plot (PC1 vs PC2)...")
plt.figure(figsize=(10, 8))

# Color by cluster labels (k=2)
scatter = plt.scatter(pc1, pc2, c=labels_k2, cmap='viridis', 
                      alpha=0.3, s=10, edgecolors='none')

plt.xlabel(f'PC1 ({var_pc1:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({var_pc2:.1f}% variance)', fontsize=12)
plt.title('EEG Data in 2D PCA Space (Colored by k=2 Clusters)', fontsize=14)

# Add colorbar for cluster labels
cbar = plt.colorbar(scatter, label='Cluster Label')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Cluster 0', 'Cluster 1'])

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('week6_day3_pca_scatter_k2.png', dpi=150)
print("Plot saved: week6_day3_pca_scatter_k2.png")

# Also create a version without cluster coloring (just density)
plt.figure(figsize=(10, 8))
plt.scatter(pc1, pc2, alpha=0.1, s=5, color='steelblue', edgecolors='none')
plt.xlabel(f'PC1 ({var_pc1:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({var_pc2:.1f}% variance)', fontsize=12)
plt.title('EEG Data in 2D PCA Space (Density View)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('week6_day3_pca_density.png', dpi=150)
print("Plot saved: week6_day3_pca_density.png")

print("\nTask 1 complete: 2D PCA plots created.")
