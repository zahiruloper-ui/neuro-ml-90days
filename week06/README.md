# Week 6 — Unsupervised Learning for Neuroscience (EEG)

## Overview
Applied PCA + k-means clustering to 117 seconds of continuous EEG data (14,980 time samples, 14 EEG channels). Main finding: the data forms a single, continuous cloud in PCA space with no clear subgroups. K-means produced degenerate results across all tested k values.

## Dataset
- **EEG Eye State** (OpenML): 14,980 samples × 14 EEG channel features (V1–V14)
- Time samples from 117 seconds of continuous EEG recording
- Labels available (eye open/closed) but not used for unsupervised analysis

## Methods
1. **Preprocessing:** Selected EEG features only, standardized (mean=0, std=1)
2. **PCA:** Reduced 14D → 5D (first 5 PCs capture 95% variance)
3. **Clustering:** Tested k-means k=2 to k=10, used elbow method
4. **Visualization:** 2D PCA scatter plots (PC1 vs PC2)
5. **Interpretation:** Wrote limitations and mini report with cautious language

## Key Results

### PCA (5 PCs = 95% variance)
| PC | Variance | Top Features |
|----|----------|--------------|
| PC1 | 29.25% | V13 (+0.458), V9 (+0.456), V1 (+0.454) |
| PC2 | 23.90% | V7 (+0.489), V4 (+0.489), V5 (+0.423) |
| PC3 | 19.59% | V12 (+0.511), V6 (−0.422), V14 (−0.379) |
| PC4 | 18.78% | V8 (+0.522), V10 (+0.521), V6 (+0.347) |
| PC5 | 3.54% | V5 (+0.568), V2 (−0.523), V11 (+0.349) |

### Clustering (all degenerate)
- **k=2:** 14,979 vs 1 (99.99% in one cluster)
- **k=3:** 14,978 vs 1 vs 1 (99.9% in one cluster)
- **k=6:** 14,101 vs 875 vs 1 vs 1 vs 1 vs 1 (94% + 6% + outliers)

### Visualization
PC1–PC2 scatter plot: "tiny smudge near 0" (tight, continuous cloud, no blobs). One color dominates (99.99%), with 1–2 outlier points.

## Interpretation
No discrete brain states in this 117-second recording. EEG variation is continuous, not categorical. This is consistent with ongoing EEG (mixture of overlapping neural processes).

## Limitations
1. Clusters ≠ ground truth (k-means always partitions)
2. Time samples, not independent subjects (14,980 time points from one subject)
3. No label validation (did not compare to eye state)
4. Short recording (117 seconds may miss state transitions)
5. Single subject (may not generalize)

## Files Produced
| File | Description |
|------|-------------|
| `week6_day1_standardize_pca.py` | Preprocessing + PCA |
| `week6_day2_kmeans.py` | Clustering + elbow |
| `week6_day3_visualize.py` | 2D PCA plots |
| `week6_day6_report.md` | Mini report |
| `week6_X_reduced_5pc.joblib` | 5-PC data |
| `week6_pca_model.joblib` | PCA model |
| `week6_cluster_labels_*.joblib` | Cluster labels |
| `week6_day*.png` | 4 plots |
| `week6_notes.md` | Full notes |
| `week6_day7_final_summary.md` | Summary |

## How to Reproduce
```bash
.venv\Scripts\python.exe week6_day1_standardize_pca.py
.venv\Scripts\python.exe week6_day2_kmeans.py
.venv\Scripts\python.exe week6_day3_visualize.py
```

## Lessons Learned
1. Standardization essential before PCA/k-means
2. PCA reveals intrinsic dimensionality
3. Degenerate clustering = continuous data (informative)
4. Visualization validates clustering results
5. Clusters are patterns, not truth (report with caution)
6. Time samples ≠ subjects (clustering correlations, not people)
7. Unsupervised learning = discovery, not confirmation

## Next Steps
- Compare clusters to eye state labels (supervised validation)
- Try HMMs or time-series clustering (temporal structure)
- Test longer recordings or multiple subjects
- Week 7: Time-series classification (sliding windows + features)

