# Day 2 — NumPy Indexing & Slicing

## 1D Slicing
- Syntax: `arr[start:stop:step]` — stop is exclusive
- Negative step reverses direction: `arr[::-1]`
- Out-of-bounds slices clip silently (no error); direct OOB index raises `IndexError`
- `arr[5:2]` → empty `[]`; need `arr[5:2:-1]` to walk backwards

## 2D Slicing
- Syntax: `matrix[row_slice, col_slice]` — two independent 1D slices
- `matrix[2, :]` → full row 2 as 1D array shape `(4,)`
- `matrix[2:3, :]` → full row 2 preserved as 2D shape `(1, 4)`
- `matrix[::-1, :]` flips vertically; `matrix[:, ::-1]` flips horizontally

## Boolean Masks
- `arr > 10` returns a boolean array of same shape
- Combine with `&` (AND), `|` (OR), `~` (NOT) — always use parentheses per condition
- Never use Python `and` / `or` — they don't work element-wise
- 2D boolean mask always returns flat 1D result
- `np.where(cond, x, y)` — preserves shape, replaces values conditionally

## Fancy Indexing
- Pass an integer list/array: `arr[[0, 2, 5]]` — picks specific positions
- Duplicates allowed: `arr[[3, 3, 1]]`
- On 2D: `matrix[[0,2], [1,3]]` pairs indices → NOT a submatrix
- Use `np.ix_([rows], [cols])` to get a true submatrix grid

## View vs Copy
| Method | Type |
|---|---|
| Basic slice `arr[1:4]` | View — modifying affects original |
| Fancy index `arr[[1,2,3]]` | Copy — independent |
| Boolean mask `arr[arr>2]` | Copy — independent |
| `.copy()` | Explicit copy — always safe |

- Check with `arr.base` — returns source array if view, `None` if copy

