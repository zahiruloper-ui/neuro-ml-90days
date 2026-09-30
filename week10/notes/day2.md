# Day 2 — Generating Embeddings

## What is an embedding?

An embedding is a numerical vector representation of text. A pre-trained language model converts a sentence or document into a vector whose values collectively represent patterns of meaning.

The individual values do not have simple labels. They work together to represent relationships between concepts.

## Model used

The project uses:

```text
all-MiniLM-L6-v2
```

This model is available through the `sentence-transformers` library. It is relatively small and fast, making it suitable for a local semantic-search project.

## Generated embeddings

The corpus contains 20 abstracts. The saved embedding array has shape:

```text
(20, 384)
```

This means:

- There are 20 rows.
- Each row represents one abstract.
- Each abstract has a 384-dimensional vector.

The embeddings were saved to:

```text
week-10-embeddings/embeddings.npy
```

The data type is:

```text
float32
```

This stores the numerical values efficiently.

## Alignment

The embedding rows must remain aligned with the CSV rows:

- CSV row 0 corresponds to embedding row 0.
- CSV row 1 corresponds to embedding row 1.
- The same pattern continues for every abstract.

If the number of rows did not match, the search system might return the correct vector but display the wrong title or abstract.

## Validation

The embedding file passed these checks:

- 20 abstracts and 20 embedding rows.
- Embedding dimension is 384.
- No `NaN` values.
- No infinite values.
- The embedding file can be loaded independently.
- The embedding file is aligned with the corpus.

## Why save embeddings?

Generating embeddings requires loading the model and processing every document. Saving the embeddings means this work only needs to happen once.

The search script can later load `embeddings.npy` directly and compare a query embedding against the stored vectors. This makes repeated searches much faster.

## Important limitation

The embedding dimensions are learned representations, not manually interpretable features. A single dimension should not be treated as representing one specific concept such as EEG, memory, or disease.