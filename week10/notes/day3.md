# Day 3 — Semantic Search

## Search pipeline

The semantic-search pipeline is:

```text
user query
→ query embedding
→ cosine similarity with document embeddings
→ sort scores
→ return top-K abstracts
```

The query must be embedded with the same model used for the stored abstracts:

```text
all-MiniLM-L6-v2
```

## Cosine similarity

Cosine similarity measures the alignment between two vectors.

Conceptually:

\[
\text{cosine similarity}(a,b)
=
\frac{a \cdot b}{\|a\|\|b\|}
\]

A higher score generally means that the query and document embeddings are more semantically aligned.

A similarity score is not a probability. For example, a score of `0.8120` does not mean there is an 81.2% probability that the document is relevant.

## Similarity shapes

For one query and 20 documents:

```text
Query embedding: (1, 384)
Document embeddings: (20, 384)
Similarity matrix: (1, 20)
Similarity scores: (20,)
```

The final score array contains one score for each abstract.

## Interactive search

The search script is:

```text
week10/semantic_search.py
```

It:

- Loads the CSV corpus.
- Loads `embeddings.npy`.
- Embeds a user query.
- Computes cosine similarities.
- Sorts abstracts by score.
- Displays the requested number of results.

The `top_k` value controls how many results are shown. If the user requests more results than exist in the corpus, the script limits the output to the total number of abstracts.

## Semantic search versus keyword search

Keyword search checks whether exact words appear in a document.

Semantic search compares the meaning represented by embeddings. It can find related documents even when they use different wording.

Semantic search is useful for broad conceptual queries. Keyword search is safer when exact terms matter, such as specific genes, proteins, mutations, drugs, or experimental methods.

## Limitations

- Similarity scores are relative ranking values, not probabilities.
- Results depend on the embedding model.
- Results also depend on the topics represented in the corpus.
- A relevant document may be missed if it uses very different language.
- An apparently related document may rank highly without containing the exact term required.
