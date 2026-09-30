# Day 4 — Exploring Semantic Search

## Purpose of exploration

Testing multiple queries helps determine whether semantic-search results are reasonable and reveals the limitations of the model and corpus.

Useful query types include:

- Direct queries.
- Paraphrased queries.
- Broad queries.
- Ambiguous queries.
- Unsupported queries.

## Direct and paraphrased queries

The direct query `EEG artifact removal` and the paraphrased query `cleaning noisy electrical recordings from the brain` both ranked `Artifact removal in EEG recordings` first.

The two searches shared two of their top five results:

- Artifact removal in EEG recordings.
- Preprocessing pipelines for electroencephalography.

The top-five overlap was:

\[
\frac{2}{5} = 0.40
\]

This shows that semantic search can preserve important results across different wording, but the exact ranking can still change.

## Ambiguous queries

The query `brain imaging and neural networks` retrieved abstracts about fMRI, visual cortex representations, and brain networks.

The phrase `neural networks` can refer to:

- Machine-learning neural networks.
- Biological networks in the brain.

The model and corpus may interpret the phrase differently from the user.

## Unsupported queries

The corpus does not contain meaningful coverage of gut microbiome research. The query `gut microbiome and brain health` still returned results because the system always ranks all documents.

Its maximum similarity was `0.4051`, compared with `0.8203` for the supported query about machine learning and neurodegenerative disease.

The returned gut microbiome results were the closest available neuroscience documents, not genuine matches.

## Thresholds

A search tool could reject weak matches:

```text
if maximum similarity < threshold:
    report that no strong match was found
```

However, thresholds are not universal. They depend on the embedding model, corpus, document type, and query style.

Before choosing a threshold, collect labelled queries with known relevant and irrelevant documents. Then evaluate precision, recall, and retrieval performance at different threshold values.

## Search errors

A high-ranked document is not automatically relevant. It may simply be the least unrelated document in a narrow corpus.

A literature-search tool must balance:

- False positives: returning irrelevant documents.
- False negatives: failing to return relevant documents.

The acceptable balance depends on the application.