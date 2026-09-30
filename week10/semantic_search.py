from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
SearchResult: TypeAlias = tuple[int, str, str, float]


def load_corpus(path: Path) -> pd.DataFrame:
    """Load the abstract corpus from a CSV file."""
    return pd.read_csv(path)


def load_embeddings(path: Path) -> np.ndarray:
    """Load document embeddings from a NumPy file."""
    return np.load(path)


def load_model(model_name: str) -> SentenceTransformer:
    """Load a pre-trained sentence-embedding model."""
    return SentenceTransformer(model_name)


def search(
    query: str,
    model: SentenceTransformer,
    document_embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    top_k: int,
) -> list[SearchResult]:
    """Return the top-k documents most similar to a query."""
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if len(document_embeddings) != len(dataframe):
        raise ValueError(
            "The number of embeddings must match the number of documents."
        )

    top_k = min(top_k, len(dataframe))

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results: list[SearchResult] = []

    for index in ranked_indices:
        row = dataframe.iloc[index]
        results.append(
            (
                int(index),
                str(row["title"]),
                str(row["abstract"]),
                float(similarity_scores[index]),
            )
        )

    return results

def display_results(query: str, results: list[SearchResult]) -> None:
    """Print search results to the console."""
    print(f"\nQuery: {query}")
    print(f"Showing top {len(results)} results:")

    for rank, (_, title, abstract, score) in enumerate(results, start=1):
        print(f"\n{rank}. {title}")
        print(f"Similarity: {score:.4f}")
        print(f"Abstract: {abstract}")


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    dataframe = load_corpus(abstracts_path)
    document_embeddings = load_embeddings(embeddings_path)
    model = load_model(MODEL_NAME)

    query = input("Enter a search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    top_k_text = input("How many results should be shown? [default: 3]: ").strip()

    if top_k_text:
        try:
            top_k = int(top_k_text)
        except ValueError:
            print("Top-K must be an integer.")
            return
    else:
        top_k = 3

    if top_k <= 0:
        print("Top-K must be greater than zero.")
        return

    top_k = min(top_k, len(dataframe))

    results = search(
        query=query,
        model=model,
        document_embeddings=document_embeddings,
        dataframe=dataframe,
        top_k=top_k,
    )

    display_results(query, results)


if __name__ == "__main__":
    main()