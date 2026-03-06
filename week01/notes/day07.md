# Day 07 — String methods + .str accessor

## .str accessor basics
- Access like `df["col"].str.upper()`, `.str.len()`, `.str[:3]` (slicing)
- Works on **Series of strings** only

## String splitting
- `.str.split(" ", expand=True)` → DataFrame (one column per split part)
- `names_split[0]` = first part, `[1]` = second part
- `.fillna()` handles missing parts gracefully

## .str.extract(regex)
- `r'^(\w+)'` = capture first word from start of string
- Parentheses `()` define what gets extracted as new column
- `regex=True` by default

## .str.contains() + .str.replace()
- `.str.contains(r'[aeiou]', case=False)` — pattern matching
- `.str.replace(r',.*', '', regex=True)` — regex replacement
- Chain multiple `.replace()` for sequential cleaning
- `regex=False` = literal string replacement

## When to use .str methods
- Text cleaning, extraction, pattern matching
- Always faster than .apply() for simple operations

