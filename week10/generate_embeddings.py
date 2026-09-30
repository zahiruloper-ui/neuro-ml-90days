from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_abstract_texts(path: Path) -> list[str]:
    """Load abstract text from a CSV file."""
    dataframe = pd.read_csv(path)
    return dataframe["abstract"].fillna("").tolist()


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate one embedding vector for each text."""
    if not texts:
        raise ValueError("At least one text document is required.")

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Texts cannot be empty.")

    return model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    texts = load_abstract_texts(abstracts_path)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} abstracts...")
    embeddings = generate_embeddings(texts, model)

    np.save(embeddings_path, embeddings)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")


if __name__ == "__main__":
    main()