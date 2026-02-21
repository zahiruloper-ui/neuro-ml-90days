# Day 03 — Missing values + types + quick EDA (Exploratory Data Analysis)


## What I practiced
- Missing scan: `df.isna().sum()` (counts), `df.isna().mean()` (fractions/percent)
- Filling: `df.fillna({...})` with column-specific values (e.g., median for numeric, "Unknown" for categorical) 
- Dropping: `df.dropna(subset=[...])` to remove rows missing key columns 
- Types: `astype("string")`, `astype("boolean")`
- Quick EDA: `df.info()` to check non-null counts + dtypes, `describe()` for numeric summary, `describe(include="all")` for mixed columns 
- Frequency tables: `value_counts(dropna=False)` to include NaNs when needed 

## Key takeaways 
- `fillna` is safer when you choose fill values per column (dict style) instead of one value for all columns. 
- `dropna(subset=...)` is for “these columns are required”; keep the rest flexible. 
- If a True/False column can contain missing values, cast it to pandas nullable `boolean` dtype (not `object`). 


## Gotchas I hit
- If I don’t cast `member` to `boolean`, it can remain `object`, and `describe(include="all")` summarizes it as categorical (unique/top/freq). 

