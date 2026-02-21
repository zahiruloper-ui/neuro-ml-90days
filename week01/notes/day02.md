# Day 2 Notes

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
