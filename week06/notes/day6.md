# Week 6 Mini Report: Unsupervised Learning on EEG Data

## Summary

We applied unsupervised learning (PCA + k-means clustering) to 117 seconds of continuous EEG data (14,980 time samples, 14 EEG channels). The main finding was that the data forms a single, continuous cloud in PCA space with no clear subgroups — k-means clustering produced degenerate results (one cluster with >94% of samples) across all tested k values.

---

## Methods

We loaded the EEG Eye State dataset (OpenML), which contains 14 EEG channel features (V1–V14) recorded from one subject over 117 seconds. We standardized all features to mean=0 and std=1, then applied PCA to reduce dimensionality. The first 5 principal components captured 95% of total variance (PC1: 29%, PC2: 24%, PC3: 20%, PC4: 19%, PC5: 4%), so we reduced the data from 14 to 5 dimensions. We then ran k-means clustering with k ranging from 2 to 10, using the elbow method to identify a reasonable k.

---

## Results

PCA revealed that the first 4–5 components capture most variance, with the remaining 9–10 features contributing little unique information. However, k-means clustering produced highly degenerate results: for k=2, one cluster contained 99.99% of samples; for k=6 (elbow method suggestion), one cluster had 94% and another had 6%, with 4 clusters as single outliers. A 2D scatter plot of PC1 vs PC2 showed a "tiny smudge near 0" — a tight, continuous cloud with no visible blobs or subgroups.

---

## Interpretation

The degenerate clustering and continuous PCA cloud suggest that this EEG recording does not contain discrete, well-separated brain states. Instead, the data appears to be one continuous population with smooth variation over time. This is consistent with the nature of resting-state or ongoing EEG, which is a mixture of many overlapping neural processes rather than distinct, categorical states.

If clusters had existed, they might have corresponded to different brain states (e.g., eyes open vs. closed, alert vs. drowsy). However, the absence of clusters suggests that — at least in this 117-second recording — the subject's EEG variation was continuous, not categorical. This does not mean the data is "uninteresting"; it means the brain activity in this window was relatively homogeneous, without sharp transitions between states.

---

## Limitations and Future Directions

It is critical to emphasize that clusters are statistical patterns, not biological truth. k-means will always partition data into k groups, even if no natural clusters exist. Our degenerate results may reflect: (1) the short recording duration (117 seconds), (2) the fact that these are time samples from one subject (not independent subjects), or (3) the possibility that EEG structure in this dataset is inherently continuous.

Future work could: (1) test clustering on longer recordings or multiple subjects, (2) compare clusters to behavioral labels (e.g., eye state, task condition) if available, or (3) use alternative methods (e.g., HMMs, time-series clustering) that account for temporal structure. For now, the responsible conclusion is: **we observed one continuous EEG population with no clear subgroups in this recording**.

---

## Files and Code

- `week6_day1_standardize_pca.py` — preprocessing and PCA
- `week6_day2_kmeans.py` — clustering and elbow method
- `week6_day3_visualize.py` — 2D PCA scatter plots
- `week6_X_reduced_5pc.joblib` — 5-PC reduced data
- `week6_day3_pca_scatter_k2.png` — visualization of continuous cloud
