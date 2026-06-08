Got it—I'll keep every word exactly the same and only add Markdown structure.

***

# The 90-day roadmap (13 weeks)

## Week 1 — Python-for-ML boot

### Goal:
go from R-learner to Python beginner who can load/clean data reliably.Deliverable: Data cleaning + EDA notebook on a life-science dataset (public).

### Daily tasks:

- D1: Python basics (variables, functions, lists/dicts) + 15 tiny exercises.

- D2: NumPy arrays + broadcasting (just enough).

- D3: pandas load → inspect → clean (missing values, types).

- D4: plotting (hist, scatter, boxplot).

- D5: write 10 flashcards (“What is a DataFrame?”, “What is leakage?” etc.).

- D6: split train/valid; make a baseline (mean/median or majority class).

- D7: quiz + README (dataset description + target variable idea).

## Week 2 — Problem framing (neuro style)

You’ll practice framing: desired outcome, model output type, success metrics (distinct from model metrics).

### Deliverable:
1-page “Problem Framing Doc” for a neuroscience mini project.

### Daily tasks:

- D1: pick one neuro question (examples: classify sleep stage, predict reaction time, detect seizure segments).

- D2: define inputs/labels; identify leakage risks.

- D3: choose model output: regression vs classification.

- D4: define success metrics (project metric + real-world metric).

- D5: create dataset checklist (availability, representativeness, label quality).

- D6: create project repo structure + empty pipeline script.

- D7: quiz + finalize framing doc.

## Week 3 — Regression mastery (core modeling)

MLCC covers linear regression, loss, gradient descent, and tuning.

### Deliverable (Mini‑Project #1):
“Predict a continuous neuro variable” (e.g., age from features, response amplitude, time-to-event proxy).

### Daily tasks:

- D1: learn + implement baseline regression (scikit-learn).

- D2: feature scaling + train/valid split correctly.

- D3: Ridge vs Lasso; compare MAE/RMSE.

- D4: residual plots + failure cases.

- D5: write 15 flashcards (loss, regularization, bias/variance).

- D6: package as a reproducible Pipeline.

- D7: quiz + project README with results table.

## Week 4 — Classification mastery + metrics

MLCC covers logistic regression, thresholds, confusion matrices, precision/recall/AUC.

### Deliverable (Mini‑Project #2):
“Binary classifier on bio/neuro tabular data” with threshold tuning.

### Daily tasks:

- D1: baseline logistic regression + confusion matrix.

- D2: precision/recall tradeoff; pick threshold based on goal.

- D3: handle imbalance (class weights or resampling).

- D4: calibration check (basic reliability curve if possible).

- D5: flashcards: AUC, precision vs recall, sensitivity/specificity.

- D6: error analysis: inspect 20 mistakes.

- D7: quiz + publish notebook.

## Week 5 — Trees & ensembles (strong baselines)

### Deliverable:
compare linear/logistic vs Random Forest vs Gradient Boosting on the same dataset.

### Daily tasks:

- D1: decision tree intuition + train one.

- D2: random forest + feature importance (with caveats).

- D3: gradient boosting model.

- D4: cross-validation basics.

- D5: flashcards: bagging vs boosting, overfitting signs.

- D6: pick best model; document why (metrics + interpretability).

- D7: quiz + results table.

## Week 6 — Unsupervised learning for neuroscience

### Deliverable:
clustering + PCA notebook on a neuro-related dataset (e.g., subject embeddings/features), with interpretation.

### Daily tasks:

- D1: standardize features; PCA.

- D2: k-means clustering; choose k (simple heuristic).

- D3: visualize clusters + interpret.

- D4: write limitations (clustering isn’t “truth”).

- D5: flashcards: PCA, variance explained, clustering pitfalls.

- D6: mini report (what cluster might mean biologically).

- D7: quiz + publish artifact.

## Week 7 — Time-series basics (neuro signals)

### Deliverable (Mini‑Project #3):
simple time-series classification/regression pipeline (windowing + features).

### Daily tasks:

- D1: windowing signals (sliding windows) + feature extraction (mean, std, bandpower if you can).

- D2: build baseline classifier on windows.

- D3: subject-wise split vs random split (avoid leakage).

- D4: evaluate + error analysis by subject.

- D5: flashcards: leakage, autocorrelation, split strategies.

- D6: refactor into functions.

- D7: quiz + README.

## Week 8 — Intro neural nets (only what you need)

MLCC includes neural network fundamentals; do just enough to understand and run a small net.

### Deliverable:
a small MLP for tabular or windowed features; compare against your best tree model.

### Daily tasks:

- D1: learn perceptron/activations; implement tiny net (PyTorch).

- D2: training loop + track loss curves.

- D3: regularization (dropout or weight decay).

- D4: compare performance and training stability.

- D5: flashcards: epochs, batch size, overfitting in nets.

- D6: save/load model; inference function.

- D7: quiz + write “when NN helped vs didn’t”.

## Week 9 — Mini MLOps: reproducibility + tracking

MLCC’s “Production ML Systems” helps you think end-to-end.

### Deliverable:
reproducible run with pinned dependencies + experiment log.

### Daily tasks:

- D1: create requirements.txt or environment.yml.

- D2: deterministic seeds; config file.

- D3: simple experiment tracking (CSV log or MLflow if you want).

- D4: data versioning basics (at least dataset link + checksum).

- D5: flashcards: train/serve skew, monitoring concepts.

- D6: refactor project into “train.py” and “predict.py”.

- D7: quiz + repo cleanup.

## Week 10 — Embeddings (optional but useful)

MLCC includes embeddings and an intro to LLMs; we’ll use embeddings in a life-science way (paper abstracts / notes search).

### Deliverable:
semantic search over 20–50 neuroscience abstracts (even copied as your own notes + links, not full PDFs).

### Daily tasks:

- D1: collect abstracts/notes with citations/links.

- D2: create embeddings (free model) + store vectors.

- D3: implement top‑k retrieval.

- D4: evaluate with 10 test queries you write.

- D5: flashcards: embeddings, cosine similarity, retrieval.

- D6: add “failure cases” (ambiguous queries).

- D7: quiz + demo notebook.

## Week 11 — Deployment (free)

fast.ai notes you can deploy models and use free resources; we’ll do a minimal demo.

### Deliverable (Mini‑Project #4, optional):
Streamlit/Gradio app for one model (tabular or time-series features).

### Daily tasks:

- D1: pick model and define input format.

- D2: create minimal UI.

- D3: add validation + friendly error messages.

- D4: package model artifact.

- D5: flashcards: serialization, inference vs training.

- D6: deploy free (or provide “run locally” steps if deploying feels heavy).

- D7: quiz + demo instructions.

## Week 12 — Exams buffer + mastery week

### Deliverable:
“Mastery review” doc + improve one previous project significantly (not a new one).

### Daily tasks:

- D1–D4: fix weakest link (data cleaning, split leakage, metrics, documentation).

- D5: write 30 flashcards total (consolidation).

- D6: redo one quiz from Week 3–8 from memory.

- D7: polish GitHub + screenshots + clean README.

## Week 13 — Capstone planning (not building yet)

You wanted capstone later; this week is planning so you’re ready after day 90.

### Deliverable:
capstone proposal + dataset shortlist + success metrics.

### Daily tasks:

- D1: choose capstone theme (e.g., seizure detection, sleep staging, neural spike sorting features, cognitive score prediction).

- D2: list 2–3 datasets; check licenses and feasibility.

- D3: define model output + success metrics.

- D4: risk list (data quality, label noise, compute).

- D5: flashcards: success metric vs evaluation metric.

- D6: timeline for capstone (4–6 weeks).

- D7: final quiz + “Day 90 status” write-up.



If you want, I can also clean up spacing/line breaks (like that merged sentence in Week 1 goal) while still keeping wording unchanged.
