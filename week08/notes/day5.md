# Week 8 - Day 5: Recall (Neural Net Basics & MLP vs Trees)

**Date:** 2026-09-26
**Focus:** Concept review and flashcards on neural networks (MLP), activations, backpropagation, scaling, over/underfitting, and MLP vs tree-model tradeoffs for tabular EEG window features.

---

## 1. What an MLP actually is

Core structure:

- Inputs: feature vector `x` (here, 28 EEG window features: 14 means + 14 stds).
- Hidden layers: each applies a linear transform plus a bias and a nonlinear activation.
- Output layer: maps last hidden layer to logits, then applies an activation suited to the task (e.g., logistic for binary classification).

One-hidden-layer MLP:

- Hidden pre-activation: `z1 = W1 x + b1`.
- Hidden activation: `h = f(z1)` (e.g., ReLU or tanh).
- Output pre-activation: `z2 = W2 h + b2`.
- Output activation: `y_hat = g(z2)`.

Why nonlinearity matters:

- If `f` is the identity (no nonlinearity), `W2 f(W1 x) = (W2 W1) x` is just another single linear transform.
- Stacking purely linear layers gives no extra expressive power; you still have a linear model.
- Nonlinear activations (ReLU, tanh, etc.) allow the network to approximate complex, nonlinear decision boundaries.

---

## 2. Activations: ReLU vs tanh

ReLU (`f(z) = max(0, z)`):

- Zero for negative inputs, linear for positive inputs.
- Sparse activations (many neurons inactive), often train faster in deep nets.
- Can cause "dying ReLUs" if many neurons get stuck at 0, but works well in practice.

Tanh (`f(z) = tanh(z)`):

- Smooth, symmetric about zero, outputs in (-1, 1).
- Encourages centered hidden representations (positive and negative values).
- Can saturate for large |z|, which slows learning, but in small networks can be effective.

In your sweeps:

- ReLU and tanh behaved differently on the EEG windows; some tanh configs gave higher balanced accuracy on the tail segment while ReLU tended to be more conservative.

---

## 3. Backpropagation and optimization

Backpropagation:

- Computes the gradient of the loss with respect to each weight using the chain rule.
- Steps:
  - Forward pass: compute `y_hat` for all inputs.
  - Compute loss (e.g., cross-entropy for classification).
  - Backward pass: propagate gradients from output back through each layer.
  - Update weights using an optimizer (gradient descent or variants).

Optimizers:

- Basic gradient descent: update weights proportionally to the negative gradient.
- Adam (used in `MLPClassifier` by default): adaptive learning rate per parameter, combines ideas from momentum and RMSProp.

Convergence behavior:

- Convergence warnings at `max_iter` mean optimization stopped while the loss was still decreasing.
- Increasing `max_iter` can reduce optimization error (better fit on train), but cannot fix distribution shift between train and test.

---

## 4. Scaling and data leakage

Feature scaling with `StandardScaler`:

- `fit` learns mean and standard deviation of each feature.
- `transform` uses those parameters to standardize data:
  - `x_scaled = (x - mean) / std`.

Leakage rule:

- Fit scalers on **training data only**.
- Apply the same transform to both training and test.
- Never fit on test data, or train/test boundaries would carry information into preprocessing.

Impact on MLP:

- MLPs are sensitive to feature scales; large-scale features dominate gradients.
- Proper scaling helps optimization and stabilizes training.

Tree models:

- RF is scale-invariant; it splits based on orderings of feature values, not their magnitudes.
- RF does not require scaling, but using scaled data does not harm it.

---

## 5. Overfitting vs underfitting in neural nets

Underfitting signs:

- Low train accuracy and low test accuracy.
- Loss still high on train; model too simple or trained for too few epochs.
- Seen in small nets at 500 iterations (e.g., `(16,), act=tanh`, train ≈ 0.80, test ≈ 0.20).

Overfitting / high capacity signs:

- Train accuracy very high (~1.0), but test accuracy modest.
- Loss on train is low; test performance plateaued.
- Seen in larger/deeper nets at 2000 iterations (e.g., `(64, 32), tanh`, train ≈ 1.0, test ≈ 0.48–0.56).

Key idea:

- Optimization error (fitting train) is different from generalization error (performing on test).
- Hyperparameter tuning can fix underfitting, but cannot fix distribution shift or a fundamentally hard test regime.

---

## 6. MLP vs Random Forest on tabular EEG windows

Random Forest strengths:

- Handles small tabular datasets well.
- Scale-invariant and robust to outliers and out-of-range inputs.
- Often strong baselines for classification on engineered features.

MLP strengths:

- Can model nonlinear interactions between features.
- With sufficient data and good tuning, can match or beat tree models on some tabular tasks.

In your Week 8 experiments:

- RF had higher average accuracy and balanced accuracy across blocked folds.
- Tuned MLP was weaker overall, but showed advantages in specific segments (tail closed windows, catching rare positive cases).
- Both models struggled in certain blocks, suggesting feature limitations, not just modeling limitations.

Trade-off takeaway:

- RF is a robust baseline for EEG window classification.
- MLP can act as a specialist complement, especially where recall on rare closed windows is important.

