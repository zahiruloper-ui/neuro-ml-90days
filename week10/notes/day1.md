# Day 1 — Collecting a Neuroscience Corpus

## Week 10 goal

This week focuses on embeddings and semantic search. The final project will represent neuroscience abstracts as numerical vectors and retrieve abstracts that are semantically similar to a user query.

## Today’s work

I created a small corpus containing 20 neuroscience abstracts.

The CSV file is:

```text
week-10-embeddings/neuro_abstracts.csv
```

It contains these columns:

- `title`: The title of each abstract.
- `abstract`: The text that will later be converted into an embedding.
- `source_url`: The source or reference URL for the record.

## Dataset validation

The corpus passed all quality checks:

- 20 abstracts are present.
- All required columns are present.
- There are no missing titles.
- There are no missing abstracts.
- There are no duplicate titles.
- Every abstract contains at least 20 words.
- The shortest abstract contains 30 words.
- The longest abstract contains 57 words.

## Why dataset quality matters

Embedding models convert text into numerical vectors. If a document is empty, duplicated, or too short, its embedding may not represent useful information.

A semantic search system also depends on corpus coverage. If the corpus does not contain documents about a topic, the system cannot return a genuinely relevant result for that topic, even if the embedding model is good.

## Abstract length

Longer abstracts may contain more concepts, context, methods, and findings. However, longer documents do not automatically receive higher similarity scores.

Semantic similarity is calculated from the relationship between the query vector and document vectors. Cosine similarity measures how aligned two vectors are, rather than simply rewarding documents with more words.

Very short abstracts may lack enough context for the model to represent their meaning reliably.

## Current limitation

The current abstracts are short teaching examples rather than verified full research abstracts. This makes the dataset easy to inspect, but the final search results may not represent the performance of a real literature-search system.

A later improvement would be to replace these examples with verified abstracts from PubMed or another public research source.