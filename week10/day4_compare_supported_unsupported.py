from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

QUERIES = [
    "machine learning for detecting neurodegenerative disease",
    "gut microbiome and brain health",
]


def main() -> None:
    project_dir = Path(__file__).parent
    dataframe = pd.read_csv(project_dir / "neuro_abstracts.csv")
    document_embeddings = np.load(project_dir / "embeddings.npy")
    model = SentenceTransformer(MODEL_NAME)

    for query in QUERIES:
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
        )

        similarity_scores = cosine_similarity(
            query_embedding,
            document_embeddings,
        )[0]

        ranked_indices = np.argsort(similarity_scores)[::-1][:TOP_K]

        print(f"\nQuery: {query}")

        for rank, index in enumerate(ranked_indices, start=1):
            title = dataframe.iloc[index]["title"]
            score = similarity_scores[index]
            print(f"{rank}. {title} — {score:.4f}")

        print(f"Maximum score: {similarity_scores.max():.4f}")
        print(f"Average score: {similarity_scores.mean():.4f}")


if __name__ == "__main__":
    main()