# Week 02 Day 04 — NumPy Universal Functions (ufuncs)

## What is a ufunc?
- A function that operates element-wise on arrays in compiled C
- No Python loop needed — 100–300x faster than list comprehensions
- Examples: np.sqrt, np.abs, np.exp, np.log, np.sin, np.where, np.clip

## Key ufuncs

| Ufunc | Use |
|---|---|
| np.sqrt / np.abs / np.square | Basic math ops |
| np.exp(x) | e^x — used in softmax |
| np.log(x) | ln(x) — used in log-loss |
| np.sin / np.cos | Trig — positional encoding |
| np.where(cond, a, b) | Vectorized if/else — ReLU |
| np.clip(x, min, max) | Clamp values — safe log, grad clipping |
| np.sum / np.mean / np.std | Aggregation |
| np.argmax / np.argmin | Index of max/min — predicted class |

## Softmax
exp_scores = np.exp(scores)
softmax = exp_scores / exp_scores.sum()   # sums to 1.0

## Log-loss (binary)
loss = -(y * np.log(p) + (1-y) * np.log(1-p))
# Always clip p first: np.clip(p, 1e-7, 1-1e-7)

## Axis rule (2D)
axis=0 → collapse rows    → result shape = (cols,)
axis=1 → collapse columns → result shape = (rows,)

## Gotchas
- np.log(0) = -inf → always clip before log
- np.sqrt(-x) = nan → use np.abs first
- argmin ties → returns first match
- time.time() rounds fast ufuncs to 0.0 → guard with if ufunc_time > 0

