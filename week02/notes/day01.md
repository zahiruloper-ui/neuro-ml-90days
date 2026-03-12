# Day 08 — NumPy Arrays

## Key Concepts

- NumPy arrays are typed, fixed-size, N-dimensional grids
- All elements share the same dtype (unlike Python lists)
- Operations are vectorized — no Python loops needed

## Array Creation
| Function | Use when... |
|---|---|
| `np.array([...])` | converting a list |
| `np.zeros(shape)` | placeholder, default float64 |
| `np.ones(shape)` | placeholder, default float64 |
| `np.arange(start, stop, step)` | integer steps, stop excluded |
| `np.linspace(start, stop, num)` | exact point count, stop included |

## Attributes
- `.shape` → tuple of dimension sizes e.g. (2, 3)
- `.ndim`  → number of axes e.g. 2
- `.dtype` → element type e.g. int64, float64
- `.size`  → total elements = product of shape

## Arithmetic
- All ops (+, -, *, /, **) are element-wise
- Scalar broadcasts to every element automatically
- Division always returns float64

## Indexing (2D)
- `m[row, col]`   → single element
- `m[0, :]`       → full row
- `m[:, 1]`       → full column
- `m[0:2, 1:3]`   → submatrix (stop excluded)
- `a[::-1]`       → reversed array

