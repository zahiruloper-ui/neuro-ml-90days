# Day 6 — Refactoring

## Purpose of refactoring

Refactoring reorganizes code to make it clearer, reusable, and easier to test without intentionally changing its behavior.

The semantic-search script was refactored into reusable functions:

- `load_corpus()`
- `load_embeddings()`
- `load_model()`
- `search()`
- `display_results()`

The embedding-generation script was also refactored into:

- `load_abstract_texts()`
- `generate_embeddings()`

## Type hints

Type hints document expected inputs and outputs. For example:

```python
def load_embeddings(path: Path) -> np.ndarray:
```

This indicates that the function expects a `Path` and returns a NumPy array.

The `SearchResult` type alias represents:

```python
tuple[int, str, str, float]
```

These values are:

1. Document index.
2. Document title.
3. Abstract text.
4. Similarity score.

## Input validation

The refactored code rejects:

- Empty queries.
- Non-positive `top_k` values.
- Mismatched numbers of documents and embeddings.
- Empty lists of texts.
- Empty or whitespace-only text.

Input validation prevents invalid data from producing confusing errors later.

## Regression testing

The fixed query was:

```text
machine learning methods for EEG classification
```

The top result remained:

```text
Machine learning methods for classifying cognitive states from EEG
```

with a similarity score of:

```text
0.7134
```

The unchanged result confirms that the refactor preserved the original search behavior.

## Embedding function tests

Two test texts produced:

```text
(2, 384)
```

Invalid inputs were rejected correctly:

- Empty list.
- Empty string.
- Whitespace-only string.

## Why refactoring matters

Reusable functions make it easier to:

- Test individual components.
- Reuse code in other scripts.
- Replace a model or data source.
- Add a user interface later.
- Find and fix errors.