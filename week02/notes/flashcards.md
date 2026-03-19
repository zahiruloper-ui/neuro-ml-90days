## Week 2 Flashcards

## What is the difference between `np.arange` and `np.linspace`?
 `arange` takes a step size (stop excluded); `linspace` takes a point count (stop included).

##  What are the four key array attributes and what does each return?
 `.shape` (dimension tuple), `.ndim` (axis count), `.dtype` (element type), `.size` (total elements).

## What is the default dtype of `np.zeros()` and `np.ones()`?
 `float64`.

##  What does `m[:, 2]` return on a 2D array?
 The entire column at index 2 — all rows, col 2.

## Why does `a * b` NOT give matrix multiplication in NumPy?
 `*` is always element-wise. Use `np.dot()` for matrix multiplication.

## What does `a[::-1]` do?
 Reverses the array by stepping backwards with step = -1.

## What does `arr[7:2:-1]` return for `arr = [10,20,30,40,50,60,70,80,90,100]`?
 `[80 70 60 50 40]` — starts at index 7, walks backward, stops before index 2.

## What is the difference between `matrix[2, :]` and `matrix[2:3, :]`?
 `matrix[2, :]` returns a 1D array shape `(4,)`; `matrix[2:3, :]` returns a 2D array shape `(1, 4)`.

## Why must you write `(arr > 5) & (arr < 20)` instead of `arr > 5 & arr < 20`?
 `&` has higher precedence than `>`, so without parentheses NumPy evaluates `5 & arr` first, producing wrong results or an error.

## What does `arr[arr > 10] = 0` do?
 Boolean mask assignment — sets every element greater than 10 to 0 in-place. This modifies the original array.

## What is the difference between `matrix[[0,2], [1,3]]` and `matrix[np.ix_([0,2], [1,3])]`?
 `[[0,2],[1,3]]` pairs indices element-wise → returns `[matrix[0,1], matrix[2,3]]` (1D, 2 values). `np.ix_` selects a full 2×2 submatrix at rows 0,2 × cols 1,3.

##  Does a basic slice return a view or a copy? How do you verify?
 A view — it shares memory with the original. Verify with `result.base` — returns the source array (not `None`) if it's a view.

## What is a NumPy ufunc?
A function that operates element-wise on arrays using compiled C code —
no Python loop needed, typically 100–300× faster.


## What does `np.where(x > 0, x, 0)` do?
Returns `x` where `x > 0`, and `0` elsewhere —
this is the ReLU activation function.


## Why do we `np.clip(probs, 1e-7, 1-1e-7)` before computing log-loss?
Because `np.log(0) = -inf` and `np.log(1-1) = -inf` —
clipping prevents a numerical crash.


## What does `np.argmax(batch, axis=1)` return for a shape `(3, 4)` array?
A shape `(3,)` array — the index of the highest value in each row,
i.e. the predicted class per sample.


## What is the softmax formula in NumPy?
`softmax = np.exp(scores) / np.exp(scores).sum()`
Converts raw logits to probabilities that sum to 1.0.


## What is the axis rule for 2D aggregation?
`axis=0` collapses rows → result shape is `(cols,)`
`axis=1` collapses columns → result shape is `(rows,)`

## `np.dot()` 1D result?  
Scalar

##  Cosine range?  
 [-1,1]

## Matrix mult shape rule?  
 Inner dims match

##  `(2,3).T` shape?  
 `(3,2)`

##  4→3 neurons `W.shape`?  
 `(4,3)`

##  Forward pass shape flow?  
 `X(b,4)@W(4,3)+b(3)→y(b,3)`

## What does np.random.seed(42) guarantee?
 Same random sequence every run with seed=42

## randint(1,6,3) generates what range? 
 Integers [1,6) → 1,2,3,4,5

## rand(3,4).shape returns?
 (3,4) - 2D array of uniform [0,1) values

## shuffle() vs permutation() difference?
 shuffle() in-place, permutation() returns copy

## Typical neural net weight init?
## normal(0, 0.01, (input_size, output_size))

## Create fake binary classification dataset (1 line each):
 X = normal(0,1,1000); y = (X > 0).astype(int)




