from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    dataframe = pd.read_csv(abstracts_path)
    embeddings = np.load(embeddings_path)

    print(f"Number of abstracts: {len(dataframe)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Embedding file size: {embeddings_path.stat().st_size} bytes")
    print(f"Contains NaN values: {np.isnan(embeddings).any()}")
    print(f"Contains infinite values: {np.isinf(embeddings).any()}")

    rows_match = len(dataframe) == embeddings.shape[0]
    expected_dimension = embeddings.shape[1] == 384

    print(f"Rows match embeddings: {rows_match}")
    print(f"Embedding dimension is 384: {expected_dimension}")

    if rows_match and expected_dimension:
        print("\nEmbedding file is aligned with the corpus.")
    else:
        print("\nEmbedding file needs investigation.")


if __name__ == "__main__":
    main()