# Week 1 flashcards

1) Q: What is a variable?
   A: Placeholder for a value

2) Q: What is a function?
   A: Performs a specific task

3) Q: What is NumPy used for?
   A: array, calculation

4) Q: What is pandas used for?
   A: dataframes

5) Q: What is a training/validation split?
   A: Split data into training (fit the model) and validation (evaluate/tune on unseen data) so you can estimate generalization and avoid overfitting (more knowledge needed)

## Day 1:

Q: What does len(xs) return? A: number of items

Q: What does sum(xs) do? A: sum of all items
​

Q: Why do we raise ValueError on empty input? A: for not doing calculation on invalid values

Q: What is min–max scaling? A: to convert values in the [0,1] scale

Q: What does math.sqrt(x) return? A: square root of a value


## Day 2:

Q: What does groupby do in pandas? A: Split data into groups by key(s) and let you summarize each group

​
Q: What does agg do after groupby? A: Applies agg functions (like min/max/mean/sum per group

Q: Why does groupby("city") often make city disappear as a column? A: Because the group key becomes the index by default

​
Q: What does reset_index() do after a groupby? A: Turns the index (group keys) back into a normal column and resets to 0,1,2... index

​
Q: Why use pd.NamedAgg? A: To give clean names to aggregated columns (avoid messy multi-level column names)​

Q: What is concat used for? A: Stack DataFrames together (often add more rows)
​
Q: What is merge used for? A: Join two DataFrames on a key column (like SQL join)

​
Q: In a left merge, what happens when there is no matching key? A: The left row stays, and the new right-side columns become NaN

## Day 3:

Q: What does df.isna().sum() return?
A: Missing-value counts per column.

Q: How do you get percent missing per column?
A: (df.isna().mean() * 100).

Q: What does df.fillna({...}) do?
A: Fills NA/NaN using per-column values (dict keys are column names).
​

Q: What does df.dropna(subset=["colA","colB"]) do?
A: Drops rows where colA or colB is missing.
​

Q: What’s the purpose of df.info() during EDA?
A: Quick check of dtypes + non-null counts per column.
​

Q: In describe(include="all"), what do top and freq mean?
A: Most common value and its count (for non-numeric/categorical columns)

## Day 4
Q: In pandas, why do we use & / | instead of and / or when combining conditions? 
​
A: Because pandas conditions are Series; &/| do elementwise boolean logic, while and/or don’t work with Series.

Q: What does df["col"].isin(["A","B"]) return?
​
A: A boolean Series mask: True where col is in the list, else False.

Q: .loc[0:3] returns which rows when index is 0..n?
​
A: Rows with labels 0,1,2,3 (stop label included).

Q: .iloc[0:3] returns which rows?
​
A: Row positions 0,1,2 (stop excluded).

Q: How do you control where NaNs appear when sorting?
​
A: Use na_position="first" or na_position="last" in sort_values().

Q: What does DataFrame.apply(func, axis=1) pass into func?
​
A: Each row as a Series (index = column names).

## Day 5

Q: What does `df.groupby("city")["score"].mean()` return?
A: A Series with city as the index and the mean score per city as values.

Q: What is the difference between `as_index=False` and `.reset_index()` after groupby?
A: Both produce the same result — the group key becomes a regular column instead of the index. `as_index=False` is set at groupby time; `.reset_index()` is chained after aggregation.

Q: How do you run multiple aggregations (count, mean, min, max) on a grouped column?
A: `df.groupby("col")["val"].agg(['count','mean','min','max'])` — returns a DataFrame with one column per function.

Q: What type of index does a multi-key groupby produce?
A: A MultiIndex (hierarchical index) with one level per groupby key.

Q: What does `df.pivot_table(values="score", index="city", columns="dept", aggfunc="mean")` do?
A: Creates a cross-tab grid where rows = unique cities, columns = unique depts, and cells = mean score. Missing combos are NaN.

Q: How do you fill missing combinations in a pivot table?
A: Pass `fill_value=0` (or any value) to `pivot_table()` — it replaces NaN for city/dept combos that don't exist in the data.

