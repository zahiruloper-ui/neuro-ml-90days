# Week 2 Day 3 — NumPy Broadcasting

## What is broadcasting?
NumPy's rule system for performing arithmetic on arrays with different shapes,
without copying data. Shapes are compared right-to-left (trailing dims first).

## The 3 Rules (right-to-left comparison)
1. **Pad left with 1s** — if ndim differs, prepend 1s to the smaller shape
   e.g. (4,) → (1,4) when paired with (3,4)
2. **Stretch size-1 dims** — any dim with size 1 is virtually expanded to match
3. **Error if incompatible** — if sizes differ and neither is 1 → ValueError

## Shape compatibility cheatsheet
| A shape  | B shape  | Compatible? | Result   |
|----------|----------|-------------|----------|
| (3, 4)   | (4,)     | ✅          | (3, 4)   |
| (3, 1)   | (5,)     | ✅          | (3, 5)   |
| (4, 1)   | (3, 1)   | ✅          | (4, 3)   |
| (3, 4)   | (3,)     | ❌          | ValueError |

## Practical patterns
- **Row normalization**: `data - data.mean(axis=1, keepdims=True)` → shape (N,1)
- **Col normalization**: `data - data.mean(axis=0, keepdims=True)` → shape (1,F)
- **Bias addition**:    `data + bias` where bias.shape=(F,) → broadcasts as (1,F)

## np.newaxis / reshape fixes
- `v[:, np.newaxis]`  → (N,) becomes (N,1)  — col vector
- `v[np.newaxis, :]`  → (N,) becomes (1,N)  — row vector
- `v.reshape(-1, 1)`  → same as np.newaxis col fix, more explicit

## ⚠️ Silent bug warning
Shape (3,) + (3,3) does NOT error — NumPy treats (3,) as (1,3) and adds
row-wise. If you wanted col-wise, you get wrong results with no error.
Always verify intent with `.shape` before broadcasting across axes.

## Key functions used
- `arr.mean(axis, keepdims=True)`
- `np.newaxis` (alias for None)
- `arr.reshape(-1, 1)`

