from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

QUERIES = [
    "EEG artifact removal",
    "cleaning noisy electrical recordings from the brain",
]


def search(
    query: str,
    model: SentenceTransformer,
    document_embeddings: np.ndarray,
    dataframe: pd.DataFrame,
) -> list[tuple[int, str, float]]:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = np.argsort(similarity_scores)[::-1][:TOP_K]

    return [
        (
            int(index),
            str(dataframe.iloc[index]["title"]),
            float(similarity_scores[index]),
        )
        for index in ranked_indices
    ]


def main() -> None:
    project_dir = Path(__file__).parent
    dataframe = pd.read_csv(project_dir / "neuro_abstracts.csv")
    document_embeddings = np.load(project_dir / "embeddings.npy")
    model = SentenceTransformer(MODEL_NAME)

    results_by_query = {}

    for query in QUERIES:
        results = search(
            query,
            model,
            document_embeddings,
            dataframe,
        )
        results_by_query[query] = results

        print(f"\nQuery: {query}")
        for rank, (_, title, score) in enumerate(results, start=1):
            print(f"{rank}. {title} — {score:.4f}")

    first_titles = {
        title for _, title, _ in results_by_query[QUERIES[0]]
    }
    second_titles = {
        title for _, title, _ in results_by_query[QUERIES[1]]
    }

    common_titles = first_titles & second_titles

    print("\nCommon results:")
    for title in sorted(common_titles):
        print(f"- {title}")

    overlap = len(common_titles) / TOP_K
    print(f"\nTop-{TOP_K} overlap: {len(common_titles)}/{TOP_K}")
    print(f"Overlap proportion: {overlap:.2f}")


if __name__ == "__main__":
    main()