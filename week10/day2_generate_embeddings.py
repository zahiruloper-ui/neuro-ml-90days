from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    dataframe = pd.read_csv(abstracts_path)

    texts = dataframe["abstract"].fillna("").tolist()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} abstracts...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    np.save(embeddings_path, embeddings)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"First vector preview: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()