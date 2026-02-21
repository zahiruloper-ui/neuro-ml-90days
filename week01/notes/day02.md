# Day 2 Notes

- What I did: Practiced pandas groupby + agg, named aggregations (pd.NamedAgg), and combining DataFrames using concat and merge

- What I learned: groupby summarizes data per group (split-apply-combine); group keys become the index by default; reset_index() brings keys back as columns; concat stacks data; merge joins on keys and left join gives NaN for unmatched keys

- Next step: Do more drills on merge types (inner/left) and practice 1–2 real mini datasets
​
- What I coded: groupby(...).agg(...) summaries (min/max/mean/sum), named aggregations (pd.NamedAgg), pd.concat(...), and pd.merge(..., how="left")

- What I learned: Left merge keeps all left rows and fills missing matches with NaN; reset_index() makes groupby results easier to merge/export


## 🔹 pd.concat()

```python
pd.concat([df1, df2], ignore_index=True) #concat stacks data (same columns) to add more rows;
                                         #ignore_index=True gives a fresh 0..n-1 index.
```
## 🔹 pd.merge()

```python
pd.merge(
    df1,                 # left DataFrame (primary table)
    df2,                 # right DataFrame (lookup table)
    on="col_name",       # matching key in both tables
    how="left"           # left join: keep all rows from df1
)
