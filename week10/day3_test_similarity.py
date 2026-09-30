from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
TEST_QUERY = "machine learning methods for EEG classification"


def main() -> None:
    project_dir = Path(__file__).parent
    abstracts_path = project_dir / "neuro_abstracts.csv"
    embeddings_path = project_dir / "embeddings.npy"

    dataframe = pd.read_csv(abstracts_path)
    document_embeddings = np.load(embeddings_path)

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [TEST_QUERY],
        convert_to_numpy=True,
    )

    similarity_matrix = cosine_similarity(
        query_embedding,
        document_embeddings,
    )

    similarity_scores = similarity_matrix[0]

    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embeddings shape: {document_embeddings.shape}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity scores shape: {similarity_scores.shape}")
    print(f"Minimum similarity: {similarity_scores.min():.4f}")
    print(f"Maximum similarity: {similarity_scores.max():.4f}")
    print(f"Contains NaN values: {np.isnan(similarity_scores).any()}")
    print(f"Contains infinite values: {np.isinf(similarity_scores).any()}")
    print(f"Titles available: {len(dataframe)}")

    if (
        query_embedding.shape == (1, 384)
        and document_embeddings.shape == (len(dataframe), 384)
        and similarity_scores.shape == (len(dataframe),)
        and not np.isnan(similarity_scores).any()
        and not np.isinf(similarity_scores).any()
    ):
        print("\nSimilarity calculation is valid.")
    else:
        print("\nSimilarity calculation needs investigation.")


if __name__ == "__main__":
    main()