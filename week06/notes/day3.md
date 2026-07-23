
## Week 6 — Day 3: Visualize (2D PCA scatter of EEG data)

### Plots Created
1. `week6_day3_pca_scatter_k2.png` — PC1 vs PC2, colored by k=2 cluster labels
2. `week6_day3_pca_density.png` — PC1 vs PC2, density view (no coloring)

### Visual Observation
- The data appears as a **tiny smudge near (0, 0)** in PC1–PC2 space.
- This is expected: after standardization, all PCs are centered at 0.
- The cloud is **continuous and compact**, with no visible subgroups or blobs.
- When colored by cluster, one color dominates (Cluster 0 = 99.99% of points), with 1–2 outlier points in a different color.

### Interpretation
- The 2D visualization confirms Day 2's finding: **no natural cluster structure** in the EEG data.
- The "smudge" shape suggests:
  - EEG variation is **smooth and continuous**, not discrete states.
  - Most samples are very similar in PC space (hence the tight cloud).
  - Outliers exist but are rare (the 1–2 differently colored points).

### Why Visualization Still Matters
- Even though clustering failed, the plot **validates** the degenerate result.
- It shows the data is **not badly preprocessed** (no weird artifacts, just a clean cloud).
- It builds intuition: if clusters existed, you'd see separate blobs or elongated structures.

### Key Takeaway
Visualization + clustering together tell a consistent story: this EEG dataset is one continuous population, not multiple subgroups. This is valuable knowledge — it tells you not to force clustering interpretations onto continuous data.