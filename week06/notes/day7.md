# Week 6 — Final Results Summary

## Workflow Completed

✅ Loaded EEG Eye State dataset (14,980 samples × 14 EEG channels)  
✅ Standardized features (mean=0, std=1)  
✅ Ran PCA: first 5 PCs capture 95% of variance  
✅ Reduced data from 14D → 5D  
✅ Tested k-means with k=2 to k=10  
✅ Created 2D PCA scatter plots (PC1 vs PC2)  
✅ Wrote limitations section on clustering pitfalls  
✅ Wrote mini report with cautious interpretation  

## Key Findings

1. **PCA reduced dimensionality effectively:** 5 PCs captured 95% of variance (PC1: 29%, PC2: 24%, PC3: 20%, PC4: 19%, PC5: 4%).
2. **Clustering was degenerate:** All k values (2–10) produced one dominant cluster (>94% of samples), indicating no natural subgroups.
3. **Visualization confirmed continuous structure:** PC1–PC2 scatter plot showed a "tiny smudge near 0" — a tight, continuous cloud with no blobs.
4. **Interpretation:** EEG variation in this 117-second recording is continuous, not categorical — no evidence of discrete brain states.

## Files Produced

| File | Description |
|------|-------------|
| `week6_day1_standardize_pca.py` | Preprocessing + PCA |
| `week6_day2_kmeans.py` | Clustering + elbow method |
| `week6_day3_visualize.py` | 2D PCA scatter plots |
| `week6_day6_report.md` | Mini report (interpretation) |
| `week6_X_reduced_5pc.joblib` | 5-PC reduced data |
| `week6_pca_model.joblib` | PCA model (for reference) |
| `week6_cluster_labels_k2.joblib` | k=2 cluster labels |
| `week6_cluster_labels_best.joblib` | k=6 cluster labels |
| `week6_day1_pca_variance.png` | PCA variance plot |
| `week6_day2_elbow_plot.png` | Elbow method plot |
| `week6_day3_pca_scatter_k2.png` | PCA scatter (colored by cluster) |
| `week6_day3_pca_density.png` | PCA scatter (density view) |
| `week6_notes.md` | Full notes (Days 1–7) |
| `week6_day7_final_summary.md` | This summary file |

## Flashcards Mastered

- #31–#35: PCA basics (standardization, explained variance, dimensionality reduction)
- #36–#40: Clustering pitfalls (degenerate results, elbow method limits)
- #41–#45: Visualization (interpreting 2D PCA plots)
- #46–#52: Limitations (clusters ≠ truth, responsible reporting)

## Final Quiz Score

___ / 10

## What I Learned

1. Standardization is essential before PCA and k-means.
2. PCA reveals intrinsic dimensionality — how many truly independent sources of variation exist.
3. Degenerate clustering is informative: it tells you the data is continuous, not clustered.
4. Visualization validates clustering results (or lack thereof).
5. Clusters are patterns, not ground truth — always report with caution.
6. Time samples ≠ independent subjects — clustering correlations, not people.
7. Unsupervised learning is for discovery, not confirmation.

## Next Steps

- Apply this workflow to a labeled dataset (e.g., compare clusters to eye state labels).
- Try time-series clustering methods (e.g., HMMs) that account for temporal structure.
- Test on longer recordings or multiple subjects to see if clusters emerge.
- Move to Week 7: [next topic — e.g., supervised learning, classification, etc.].
