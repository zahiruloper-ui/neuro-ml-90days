# Day 04 — Filtering, selecting, sorting, features, apply

## Filtering rows (boolean masks)
- Basic pattern: `df[condition]` where condition is a boolean Series with same length as df.
- Combine conditions with `&` (AND) and `|` (OR); use parentheses around each condition, e.g. `(cond1) & (cond2)`. 
- Use `col.isin([...])` for “value in list” filters. 
- Missing values: comparisons with NaN usually behave like False in a boolean mask; use `.isna()` / `.notna()` when you need explicit control. [web:12]
- `df.query("expr")` is an alternative string-based filter (nice readability for simple cases). 

## Selecting rows/cols safely (`loc` vs `iloc`)
- `.loc[row_labels_or_mask, col_labels]` is label-based; label slices are inclusive on both ends when the labels exist (e.g. `loc[0:3]` includes 3). 
- `.iloc[row_positions, col_positions]` is position-based; Python slicing rules apply (end excluded).
- Scalars: `.at[row_label, col_label]` and `.iat[row_pos, col_pos]` return a single value. 
- Safe assignment pattern: `df.loc[mask, "col"] = value` (avoids chained-indexing issues). 

## Sorting
- Use `df.sort_values(by="col")` for one column, or `by=["c1","c2"]` for multi-column sorts; `ascending` can be a list matching `by`. 
- NaN placement is controlled by `na_position="last"` (default) or `"first"`. [web:9]

## Feature creation (new columns)
- Vectorized ops are preferred: `df["new"] = df["a"] + 5`, `df["flag"] = df["score"] >= 85`. 
- Conditional feature: `np.where(condition, x, y)` builds a full-length array for assignment. 
- String features: `df["city"].str.lower()` (vectorized string methods).
- `df.assign(...)` returns a new DataFrame (useful for method chaining / pipelines). 

## apply() basics
- `df.apply(func, axis=0)` applies `func` to each column; `axis=1` applies to each row (row passed as a Series).
- Prefer vectorized solutions (`np.where`, comparisons, `.str`, etc.) over `apply(axis=1)` when possible for speed/readability. [web:212]

