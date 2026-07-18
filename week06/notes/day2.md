## Week 6 — Day 2: Cluster (k-means on EEG data)

### Key Finding: Degenerate Clustering
- **k=3:** 14,978 vs 1 vs 1 samples (99.9% in one cluster)
- **k=6 (elbow method):** 14,101 vs 875 vs 1 vs 1 vs 1 vs 1 (94% + 6% + outliers)
- **k=2:** 14,979 vs 1 sample (99.99% in one cluster)

### Interpretation
- The EEG data in 5-PC space forms **one continuous cloud** with no clear substructure.
- k-means is **forcing clusters where none naturally exist** — this is a common pattern in real data.
- The elbow method suggested k=6, but even that gave degenerate results.

### Why This Happens
1. **PCA already captured most variance** — remaining variation is continuous, not clustered.
2. **EEG data is inherently mixed** — ongoing brain activity doesn't fall into discrete "states" in this dataset.
3. **k-means assumes spherical clusters** — if data is a continuous cloud, k-means will still partition it, but the partitions are artificial.

### Files Saved
- `week6_cluster_labels_k2.joblib` — k=2 labels (degenerate)
- `week6_cluster_labels_best.joblib` — k=6 labels (still degenerate)
- `week6_day2_elbow_plot.png` — elbow curve
