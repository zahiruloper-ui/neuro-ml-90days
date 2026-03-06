# Day 06 — Pivot mastery + crosstab + apply intro

## pivot_table margins=True
- Adds "All" row + "All" column with subtotals
- `All` column = row averages, `All` row = column averages
- NaN cells stay NaN in margins

## pd.crosstab()
- `pd.crosstab(df["row_var"], df["col_var"])` — counts by default
- `margins=True` — add totals
- `normalize='index'` — row proportions (each row sums to 1)
- `normalize='columns'` — column proportions
- `normalize='all'` — table proportions

## .apply() row-wise (axis=1)
- `df.apply(function, axis=1)` — runs once per row
- Input = row Series: `row["col"]` extracts column values
- Use for complex row logic that vectorized ops can't do
- Lambda version: `df.apply(lambda row: logic, axis=1)`

## apply() best practices
- Prefer vectorized (`np.where`, `pd.cut`) when possible — faster
- Use apply for: multiple conditions, string logic, custom row transforms

