"""Common EEG windowed feature pipeline and model utilities for Week 8.

This module centralizes:
- loading the EEG Eye State CSV
- creating sliding windows
- extracting mean/std features
- building blocked time-series splits with buffer
- training and evaluating RF and MLP models on given splits
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE_DEFAULT = 128
STEP_DEFAULT = 64


@dataclass
class DatasetConfig:
    csv_path: str = "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
    window_size: int = WINDOW_SIZE_DEFAULT
    step: int = STEP_DEFAULT


@dataclass
class SplitConfig:
    n_blocks: int = 5
    buffer: int = 1


@dataclass
class ModelConfig:
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    rf_random_state: int = 42
    mlp_hidden: tuple[int, ...] = (64, 32)
    mlp_activation: str = "tanh"
    mlp_max_iter: int = 500
    mlp_random_state: int = 42


@dataclass
class FoldResult:
    fold: int
    rf_acc: float
    rf_bal_acc: float
    mlp_acc: float
    mlp_bal_acc: float
    rf_cm: np.ndarray
    mlp_cm: np.ndarray


def load_eeg_dataset(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load EEG Eye State CSV and return (X_raw, y_raw)."""
    df = pd.read_csv(cfg.csv_path)
    label_col = "eyeDetection" if "eyeDetection" in df.columns else df.columns[-1]
    X_raw = df.drop(columns=[label_col]).values.astype(float)
    y_raw = df[label_col].values.astype(int)
    return X_raw, y_raw


def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE_DEFAULT,
    step: int = STEP_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and majority labels per window."""
    Xw, yw = [], []
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        Xw.append(X[start:end])
        yw.append(int(y[start:end].mean() > 0.5))
    return np.array(Xw), np.array(yw)


def extract_features(Xw: np.ndarray) -> np.ndarray:
    """Mean + std features per window (axis=1)."""
    return np.hstack([Xw.mean(axis=1), Xw.std(axis=1)])


def make_block_indices(n_windows: int, n_blocks: int) -> list[np.ndarray]:
    """Return list of contiguous window index blocks for blocked CV."""
    indices = np.arange(n_windows)
    return list(np.array_split(indices, n_blocks))


def blocked_time_splits(
    n_windows: int,
    block_indices: list[np.ndarray],
    buffer: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs for blocked CV with buffer."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n_windows)
    for b in block_indices:
        min_t, max_t = b[0], b[-1]
        test_idx = b
        mask = (all_idx < min_t - buffer) | (all_idx > max_t + buffer)
        train_idx = all_idx[mask]
        splits.append((train_idx, test_idx))
    return splits


def build_models(cfg: ModelConfig) -> tuple[RandomForestClassifier, MLPClassifier]:
    """Instantiate RF and MLP according to ModelConfig."""
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.rf_random_state,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden,
        activation=cfg.mlp_activation,
        solver="adam",
        max_iter=cfg.mlp_max_iter,
        random_state=cfg.mlp_random_state,
    )
    return rf, mlp


def run_blocked_cv(
    X_features: np.ndarray,
    y_windows: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    cfg: ModelConfig,
) -> list[FoldResult]:
    """Run RF and MLP across blocked splits and return per-fold results."""
    rf, mlp = build_models(cfg)
    results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X_features[train_idx], y_windows[train_idx]
        X_test, y_test = X_features[test_idx], y_windows[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf.fit(X_train, y_train)
        mlp.fit(X_train_s, y_train)

        rf_pred = rf.predict(X_test)
        mlp_pred = mlp.predict(X_test_s)

        rf_acc = accuracy_score(y_test, rf_pred)
        mlp_acc = accuracy_score(y_test, mlp_pred)
        rf_bal = balanced_accuracy_score(y_test, rf_pred)
        mlp_bal = balanced_accuracy_score(y_test, mlp_pred)

        rf_cm = confusion_matrix(y_test, rf_pred)
        mlp_cm = confusion_matrix(y_test, mlp_pred)

        results.append(
            FoldResult(
                fold=fold,
                rf_acc=rf_acc,
                rf_bal_acc=rf_bal,
                mlp_acc=mlp_acc,
                mlp_bal_acc=mlp_bal,
                rf_cm=rf_cm,
                mlp_cm=mlp_cm,
            )
        )

    return results


def summarize_results(results: list[FoldResult]) -> tuple[float, float, float, float]:
    """Return mean accuracies and balanced accuracies for RF and MLP."""
    rf_accs = [r.rf_acc for r in results]
    rf_bals = [r.rf_bal_acc for r in results]
    mlp_accs = [r.mlp_acc for r in results]
    mlp_bals = [r.mlp_bal_acc for r in results]
    return (
        float(np.mean(rf_accs)),
        float(np.mean(rf_bals)),
        float(np.mean(mlp_accs)),
        float(np.mean(mlp_bals)),
    )

