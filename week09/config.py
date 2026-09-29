import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv"
OUT_DIR = Path("week09/data")
OUT_PATH = OUT_DIR / "eeg-eye-state.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {URL}")
    urllib.request.urlretrieve(URL, OUT_PATH)
    print(f"Saved to: {OUT_PATH}")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"File size: {size_kb:.1f} KB")

    df = pd.read_csv(OUT_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()

from datetime import datetime

SEED = 42

DATA_PATH = "week09/data/eeg-eye-state.csv"

WINDOW_SIZE = 448
STEP = 64

N_BLOCKS = 5
BUFFER = 1

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": SEED,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "tanh",
    "max_iter": 500,
    "random_state": SEED,
}

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005

if __name__ == "__main__":
    print(f"RUN_ID: {RUN_ID}")
    print(f"SEED: {SEED}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}, STEP: {STEP}")
    print(f"N_BLOCKS: {N_BLOCKS}, BUFFER: {BUFFER}")
    print(f"RF_PARAMS: {RF_PARAMS}")
    print(f"MLP_PARAMS: {MLP_PARAMS}")