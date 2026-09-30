from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    dataframe = pd.read_csv(abstracts_path)
    document_embeddings = np.load(embeddings_path)
    model = SentenceTransformer(MODEL_NAME)

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

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = np.argsort(similarity_scores)[::-1]

    print(f"\nQuery: {query}")
    print(f"Showing top {top_k} results:")

    for rank, index in enumerate(ranked_indices[:top_k], start=1):
        title = dataframe.iloc[index]["title"]
        abstract = dataframe.iloc[index]["abstract"]
        score = similarity_scores[index]

        print(f"\n{rank}. {title}")
        print(f"Similarity: {score:.4f}")
        print(f"Abstract: {abstract}")


if __name__ == "__main__":
    main()