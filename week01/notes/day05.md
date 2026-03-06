# Day 05 — Aggregation

## groupby basics
- `df.groupby("col")["val"].mean()` — single key, single aggregation
- Result is a Series with the group key as index
- `as_index=False` or `.reset_index()` keeps group key as a regular column

## .agg() with multiple functions
- `df.groupby("col")["val"].agg(['count','mean','min','max'])`
- Returns a DataFrame; each function becomes a column
- Can also pass a dict for named output: `.agg(avg=("score","mean"))`

## Multi-key groupby
- `df.groupby(["col1","col2"])["val"].mean()` — result has MultiIndex
- Chain `.sort_values(ascending=False)` to sort by aggregated value
- Chain `.reset_index()` to flatten MultiIndex into a regular DataFrame

## pivot_table
- `df.pivot_table(values=, index=, columns=, aggfunc=)`
- `index=` → rows, `columns=` → columns, `values=` → what to aggregate
- Missing combos become NaN; use `fill_value=0` to replace them
- Best for cross-tab / reporting grids; groupby better for chaining

