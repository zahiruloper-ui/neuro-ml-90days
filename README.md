# 90-Day ML Learning Progression (Life Sciences Focus)

A structured, self-directed learning path from Python fundamentals through classification, clustering, time-series, and intro deep learning — applied to life-science and neuroscience datasets.

This repo documents real progress, not a fixed schedule: some weeks move faster, some slower, and the tracker below reflects where things actually stand.

## How to Read This Repo

- Each week has its own folder (`week-01-python-boot/`, `week-02-problem-framing/`, etc.) containing the notebook(s) and a short `README.md` with the deliverable and key results.
- Daily task breakdowns and flashcards are kept in each week's folder under `daily-log.md` — useful for me, optional reading for visitors.
- The table below is the single source of truth for progress. It's updated as weeks are completed, not on a fixed calendar.

## Progress Tracker

| Week | Focus | Deliverable | Status |
|------|-------|-------------|--------|
| 1 | Python-for-ML boot | Data cleaning + EDA notebook (life-science dataset) | ✅ Done |
| 2 | Problem framing | 1-page Problem Framing Doc (neuro mini project) | ✅ Done |
| 3 | Regression mastery | Mini-Project #1: predict a continuous neuro variable | ✅ Done |
| 4 | Classification + metrics | Mini-Project #2: binary classifier w/ threshold tuning | ✅ Done |
| 5 | Trees & ensembles | Linear vs Random Forest vs Gradient Boosting comparison | ✅ Done |
| 6 | Unsupervised learning | Clustering + PCA notebook with biological interpretation | ✅ Done |
| 7 | Time-series basics | Mini-Project #3: windowed signal classification/regression | 🔄 In Progress |
| 8 | Intro neural nets | Small MLP vs best tree model comparison | ⬜ Planned |
| 9 | Mini MLOps | Reproducible run: pinned deps + experiment log | ⬜ Planned |
| 10 | Embeddings | Semantic search over neuroscience abstracts | ⬜ Planned |
| 11 | Deployment | Mini-Project #4 (optional): Streamlit/Gradio demo app | ⬜ Planned |
| 12 | Mastery review | Improve one prior project significantly | ⬜ Planned |
| 13 | Capstone planning | Capstone proposal + dataset shortlist | ⬜ Planned |

> Update statuses to 🔄 In Progress / ✅ Done as you complete each week — keep the header note "Currently working on: Week X" in sync too.

**Currently working on: Week 7**

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib`/`seaborn` · `PyTorch` (Week 8+) · `Streamlit`/`Gradio` (Week 11)

## Repo Structure

```text
.
├── README.md
├── week-01-python-boot/
│   ├── README.md
│   ├── eda-notebook.ipynb
│   └── daily-log.md
├── week-02-problem-framing/
│   ├── README.md
│   ├── problem-framing-doc.md
│   └── daily-log.md
├── week-03-regression/
│   ├── README.md
│   ├── mini-project-1-regression.ipynb
│   └── daily-log.md
├── week-04-classification/
│   ├── README.md
│   ├── mini-project-2-classifier.ipynb
│   └── daily-log.md
├── week-05-trees-ensembles/
│   ├── README.md
│   ├── model-comparison.ipynb
│   └── daily-log.md
├── week-06-unsupervised/
│   ├── README.md
│   ├── pca-kmeans-notebook.ipynb
│   └── daily-log.md
├── week-07-time-series/          # in progress
├── week-08-neural-nets/          # planned
├── week-09-mlops/                # planned
├── week-10-embeddings/           # planned
├── week-11-deployment/           # planned
├── week-12-mastery-review/       # planned
└── week-13-capstone-planning/    # planned
```

## Weekly Detail

### Week 1 — Python-for-ML Boot ✅
**Goal:** Go from R-learner to Python beginner who can load and clean data reliably.
**Deliverable:** Data cleaning + EDA notebook on a public life-science dataset.

### Week 2 — Problem Framing (Neuro Style) ✅
**Goal:** Practice framing desired outcome, model output type, and success metrics (distinct from model metrics).
**Deliverable:** 1-page "Problem Framing Doc" for a neuroscience mini project.

### Week 3 — Regression Mastery ✅
**Deliverable (Mini-Project #1):** Predict a continuous neuro variable (e.g., age from features, response amplitude, time-to-event proxy).

### Week 4 — Classification Mastery + Metrics ✅
**Deliverable (Mini-Project #2):** Binary classifier on bio/neuro tabular data with threshold tuning.

### Week 5 — Trees & Ensembles ✅
**Deliverable:** Compare linear/logistic regression vs Random Forest vs Gradient Boosting on the same dataset.

### Week 6 — Unsupervised Learning for Neuroscience ✅
**Deliverable:** Clustering + PCA notebook on a neuro-related dataset with biological interpretation.

### Week 7 — Time-Series Basics (Neuro Signals) 🔄
**Deliverable (Mini-Project #3):** Time-series classification/regression pipeline with windowing and feature extraction.

### Week 8 — Intro Neural Nets ⬜
**Deliverable:** Small MLP for tabular/windowed features, compared against best tree model.

### Week 9 — Mini MLOps: Reproducibility + Tracking ⬜
**Deliverable:** Reproducible run with pinned dependencies and an experiment log.

### Week 10 — Embeddings ⬜
**Deliverable:** Semantic search over 20-50 neuroscience abstracts (notes + citations).

### Week 11 — Deployment (Optional) ⬜
**Deliverable (Mini-Project #4):** Streamlit/Gradio app for one model.

### Week 12 — Exam Buffer + Mastery Week ⬜
**Deliverable:** Mastery review doc; significant improvement of one prior project.

### Week 13 — Capstone Planning ⬜
**Deliverable:** Capstone proposal, dataset shortlist, and success metrics for a 4-6 week capstone.
