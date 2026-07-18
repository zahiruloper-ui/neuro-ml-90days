## Week 6 — Day 1: Reduce (PCA on EEG data)

### Dataset
- **EEG Eye State** (OpenML)
- 14,980 samples × 14 EEG channel features (V1–V14)
- Time samples from 117 seconds of continuous EEG recording

### Preprocessing
1. **Selected EEG features only** (dropped eye state label)
2. **Standardized features** using `StandardScaler`
   - Mean ≈ 0, Std = 1 (to machine precision)
   - Critical because PCA is sensitive to feature scale

### PCA Results
- **Full PCA (14 components):**
  - PC1: 29.25% variance
  - PC2: 23.90% variance
  - PC3: 19.59% variance
  - PC4: 18.78% variance
  - PC5: 3.54% variance
  - **Top 5 PCs capture 95.06% of total variance**
  - PCs 11–14 capture ~0% variance redundant features

- **Reduced data:** (14980, 14) → (14980, 5)
  - Kept 95% of information with 64% fewer features

### Feature Loadings (Top 3 per PC)
| PC | Variance | Top Features (loadings) |
|----|----------|-------------------------|
| PC1 | 29.2% | V13 (+0.458), V9 (+0.456), V1 (+0.454) |
| PC2 | 23.9% | V7 (+0.489), V4 (+0.489), V5 (+0.423) |
| PC3 | 19.6% | V12 (+0.511), V6 (−0.422), V14 (−0.379) |
| PC4 | 18.8% | V8 (+0.522), V10 (+0.521), V6 (+0.347) |
| PC5 | 3.5% | V5 (+0.568), V2 (−0.523), V11 (+0.349) |

### Key Takeaways
1. **No single EEG channel dominates** — variation is distributed across channels
2. **5 PCs sufficient** for downstream clustering (Day 2)
3. **Standardization essential** — without it, PCA would be dominated by high-variance channels

### Files Saved
- `week6_X_reduced_5pc.joblib` — reduced data for clustering
- `week6_pca_model.joblib` — PCA model (for reference)
- `week6_day1_pca_variance.png` — variance explained plot
