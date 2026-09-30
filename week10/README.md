# Week 10 - Embeddings and Semantic Search

## Goal

This project builds a small semantic-search tool over 20 neuroscience abstracts.

The project demonstrates how to:

- Represent text as numerical embeddings.
- Use a pre-trained sentence-embedding model.
- Compare embeddings with cosine similarity.
- Retrieve abstracts based on semantic meaning rather than exact keywords.

## Files

- `neuro_abstracts.csv`: The neuroscience abstract corpus.
- `generate_embeddings.py`: Generates and saves document embeddings.
- `embeddings.npy`: Saved document embeddings.
- `semantic_search.py`: Interactive semantic-search program.
- `day1_*.py` through `day4_*.py`: Daily data-collection, validation, comparison, and exploration scripts.
- `notes/`: Daily learning notes and flashcards.

## Requirements

- Python 3.11 or later.
- A virtual environment.
- `pandas`.
- `numpy`.
- `sentence-transformers`.
- `scikit-learn`.

## Setup

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install pandas numpy sentence-transformers scikit-learn
```

The project uses the pre-trained model:

```text
all-MiniLM-L6-v2
```

The first run downloads the model files. Later runs can use the local model cache.

## Generate embeddings

From the repository root, run:

```powershell
.venv\Scripts\python.exe week10\generate_embeddings.py
```

This reads `neuro_abstracts.csv`, generates one embedding per abstract, and saves the result to:

```text
week10/embeddings.npy
```

For the current corpus, the expected embedding shape is:

```text
(20, 384)
```

This means there are 20 document vectors, each with 384 dimensions.

## Run semantic search

From the repository root, run:

```powershell
.venv\Scripts\python.exe week10\semantic_search.py
```

Enter a query such as:

```text
machine learning methods for EEG classification
```

Then enter the number of results to display, such as:

```text
3
```

The program returns the highest-ranked abstracts and their cosine-similarity scores.

## How the search works

The search pipeline is:

```text
abstract text
→ document embeddings
→ saved NumPy vectors
→ query embedding
→ cosine similarity
→ ranked results
```

The query and documents must use the same embedding model. Each similarity score measures vector alignment and is used for ranking. A score is not a probability of relevance.

The CSV row order must remain aligned with the embedding row order:

- CSV row 0 corresponds to embedding row 0.
- CSV row 1 corresponds to embedding row 1.
- The same pattern continues for all documents.

## Semantic search versus keyword search

Keyword search looks for exact words or phrases.

Semantic search compares the meaning represented by text embeddings. It may retrieve relevant documents that use different wording, but it can also return broadly related documents that do not contain the exact term from the query.

## Limitations

- The corpus contains only 20 short teaching abstracts.
- The source URLs are placeholders rather than individual paper links.
- The corpus covers selected neuroscience topics rather than the entire field.
- The general-purpose embedding model may not represent specialized biomedical terminology perfectly.
- Similarity scores are ranking values, not probabilities.
- The system always returns results, even when the query topic is absent from the corpus.
- No labelled relevance dataset was used.
- No similarity threshold has been calibrated for rejecting weak matches.
- Larger documents may need to be split into chunks.
- The model is loaded each time the search script starts.

## Reproducibility

The generated `embeddings.npy` file can be recreated by running `generate_embeddings.py`.

The local `.venv` folder should not be committed to Git. It is ignored through the repository's `.gitignore` file. Commit dependency descriptions or project scripts, not the installed virtual environment itself.
'@ | Set-Content -Encoding utf8 README.md