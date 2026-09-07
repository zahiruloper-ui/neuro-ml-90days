import numpy as np
import pandas as pd


def load_eeg_data(url: str):
    """
    Load EEG Eye State dataset from a CSV URL.

    Parameters
    ----------
    url : str
        URL or path to the EEG Eye State CSV file.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_channels)
        Raw EEG channel values.
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 = eye-open, 1 = eye-closed).
    """
    df = pd.read_csv(url)
    eeg_cols = [c for c in df.columns if c != "class"]
    X = df[eeg_cols].to_numpy()
    y = df["class"].to_numpy()
    return X, y


def create_sliding_windows(X: np.ndarray, y: np.ndarray, window_size: int, step: int):
    """
    Segment a multi-channel time-series into overlapping sliding windows.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels)
        Raw time-series data.
    y : np.ndarray, shape (n_samples,)
        Per-time-point labels.
    window_size : int
        Number of time samples per window.
    step : int
        Number of samples to advance between windows (step < window_size means overlap).

    Returns
    -------
    X_windows : np.ndarray, shape (n_windows, window_size, n_channels)
        Windowed EEG segments.
    y_windows : np.ndarray, shape (n_windows,)
        Label for each window (taken from the last time point in the window).
    """
    n_samples = X.shape[0]
    windows = []
    window_labels = []

    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(X[start:end, :])
        window_labels.append(y[end - 1])

    X_windows = np.stack(windows)
    y_windows = np.array(window_labels)
    return X_windows, y_windows




def extract_features(X_windows: np.ndarray):
    """
    Extract simple time-domain features (mean, std) per channel, per window.

    Parameters
    ----------
    X_windows : np.ndarray, shape (n_windows, window_size, n_channels)
        Windowed EEG segments.

    Returns
    -------
    X_features : np.ndarray, shape (n_windows, n_channels * 2)
        Concatenated [mean_features, std_features] per window.
    """
    mean_features = X_windows.mean(axis=1)
    std_features = X_windows.std(axis=1)
    X_features = np.concatenate([mean_features, std_features], axis=1)
    return X_features


def time_block_split(X_features: np.ndarray, y_windows: np.ndarray,
                      test_fraction: float = 0.2, buffer: int = 1):
    """
    Split windowed features into train/test respecting time order,
    with a buffer zone at the boundary to avoid overlap-based leakage.

    Parameters
    ----------
    X_features : np.ndarray, shape (n_windows, n_features)
        Feature matrix, in time order.
    y_windows : np.ndarray, shape (n_windows,)
        Labels, in time order (same order as X_features).
    test_fraction : float
        Fraction of windows (from the END of the time-series) to use as test.
    buffer : int
        Number of windows to drop on each side of the train/test boundary,
        to remove any remaining overlap leakage between adjacent windows.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Time-ordered train/test splits.
    """
    n_windows = X_features.shape[0]
    split_idx = int(n_windows * (1 - test_fraction))

    train_idx = np.arange(0, split_idx - buffer)
    test_idx = np.arange(split_idx + buffer, n_windows)

    X_train, y_train = X_features[train_idx], y_windows[train_idx]
    X_test, y_test = X_features[test_idx], y_windows[test_idx]
    return X_train, y_train, X_test, y_test


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def train_and_evaluate(model, X_train, y_train, X_test, y_test, scale: bool = False):
    """
    Train a classifier and evaluate it on a held-out test set.

    Parameters
    ----------
    model : sklearn-style classifier
        Must implement .fit(), .predict(), .score().
    X_train, y_train : np.ndarray
        Training features and labels.
    X_test, y_test : np.ndarray
        Test features and labels.
    scale : bool
        If True, apply StandardScaler (fit on train, applied to both train and test).
        Use True for scale-sensitive models (e.g., logistic regression),
        False for scale-insensitive models (e.g., random forest).

    Returns
    -------
    results : dict
        Dictionary with keys: 'train_acc', 'test_acc', 'confusion_matrix'.
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "train_acc": model.score(X_train, y_train),
        "test_acc": model.score(X_test, y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return results

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    URL = "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
    X, y = load_eeg_data(URL)
    X_windows, y_windows = create_sliding_windows(X, y, window_size=128, step=64)
    X_features = extract_features(X_windows)
    X_train, y_train, X_test, y_test = time_block_split(
        X_features, y_windows, test_fraction=0.2, buffer=1
    )

    logreg = LogisticRegression(max_iter=1000)
    logreg_results = train_and_evaluate(
        logreg, X_train, y_train, X_test, y_test, scale=True
    )
    print("Logistic Regression:")
    print("  Train acc:", round(logreg_results["train_acc"], 4))
    print("  Test acc:", round(logreg_results["test_acc"], 4))
    print("  Confusion matrix:\n", logreg_results["confusion_matrix"])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_results = train_and_evaluate(
        rf, X_train, y_train, X_test, y_test, scale=False
    )
    print("\nRandom Forest:")
    print("  Train acc:", round(rf_results["train_acc"], 4))
    print("  Test acc:", round(rf_results["test_acc"], 4))
    print("  Confusion matrix:\n", rf_results["confusion_matrix"])
