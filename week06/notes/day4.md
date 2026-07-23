## Week 6 — Day 4: Caution (Limitations of Clustering)

### 1. Clusters Are Not Ground Truth

- **k-means always partitions** — it will divide data into k groups even if no natural clusters exist.
- **Degenerate clustering in our EEG data** (e.g., 99.9% in one cluster) shows that k-means was forcing structure onto a continuous cloud.
- **Clusters ≠ biological categories** — a cluster is a statistical convenience, not proof of distinct brain states or subject types.

> **Key principle:** Unsupervised learning reveals *patterns*, not *truth*. Always separate "we observed a pattern" from "this pattern is biologically real."

---

### 2. Sensitivity to Features, Scaling, and Chosen k

- **Feature selection matters:** If you included different EEG channels or added non-EEG features, the cluster structure (or lack thereof) could change.
- **Scaling is critical:** PCA and k-means are both sensitive to feature scale. Without standardization, high-variance features would dominate, potentially creating artificial clusters.
- **Choice of k is heuristic:** The elbow method suggested k=6, but that still gave degenerate results. There is no "true" k — only a reasonable choice given the data and goal.
- **PCA dimensionality choice:** We kept 5 PCs (95% variance). Keeping 2 vs 10 PCs could change clustering outcomes.

---

### 3. Why Unsupervised Structure Can Be Useful Without Being Definitive

- **Exploratory value:** Even degenerate clustering told us something important — the EEG data is one continuous population, not multiple subgroups.
- **Hypothesis generation:** If clusters *had* existed, they could suggest distinct brain states to investigate further with labeled data.
- **Data quality check:** A clean, tight cloud in PCA space (as we saw) suggests no major preprocessing artifacts — the data looks "well-behaved."
- **Dimensionality reduction:** PCA itself is useful for visualization and downstream tasks, even without clustering.

> **Takeaway:** Unsupervised methods are tools for discovery, not confirmation. They help you ask better questions, not answer them definitively.

---

### 4. Specific Caveats for Our EEG Analysis

1. **Continuous variation, not discrete states:** The "tiny smudge" in PC1–PC2 space and degenerate clustering suggest EEG variation is smooth, not categorical.
2. **Time samples, not independent subjects:** Our 14,980 rows are time points from ~117 seconds of EEG, not 14,980 independent subjects. Clustering time points may not reflect meaningful brain states.
3. **No labels to validate against:** Without eye state or behavioral labels, we can't test if clusters correspond to real conditions (e.g., eyes open vs. closed).
4. **EEG is inherently noisy:** Single-trial EEG is a mixture of many signals (brain, muscle, electrode noise). Clustering may pick up noise patterns, not neural structure.
5. **Generalizability unknown:** This is one subject's EEG. Clusters (if they existed) might not replicate across subjects or sessions.

---

### 5. Responsible Interpretation Guidelines

When reporting unsupervised results, use cautious language:

| ❌ Avoid | ✅ Prefer |
|---------|----------|
| "We found 3 brain states." | "We observed 3 clusters in the data." |
| "This proves distinct neural populations." | "This suggests possible subgroups, pending validation." |
| "The data has 2 natural groups." | "k=2 gave the most balanced partition, though structure is weak." |
| "Cluster 1 is eyes-open, Cluster 2 is eyes-closed." | "Cluster labels could be compared to eye state labels in future work." |

> **Rule of thumb:** If you wouldn't say it about a labeled analysis, don't say it about an unsupervised one.

---

### Files Referenced
- `week6_day3_pca_scatter_k2.png` — shows continuous cloud, no blobs
- `week6_day2_elbow_plot.png` — elbow method suggested k=6, but clusters still degenerate
- `week6_X_reduced_5pc.joblib` — 5-PC reduced data for future analyses
