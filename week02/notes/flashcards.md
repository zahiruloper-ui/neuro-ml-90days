## Week 2 Flashcards

**Q:** What is the difference between `np.arange` and `np.linspace`?
**A:** `arange` takes a step size (stop excluded); `linspace` takes a point count (stop included).

**Q:** What are the four key array attributes and what does each return?
**A:** `.shape` (dimension tuple), `.ndim` (axis count), `.dtype` (element type), `.size` (total elements).

**Q:** What is the default dtype of `np.zeros()` and `np.ones()`?
**A:** `float64`.

**Q:** What does `m[:, 2]` return on a 2D array?
**A:** The entire column at index 2 — all rows, col 2.

**Q:** Why does `a * b` NOT give matrix multiplication in NumPy?
**A:** `*` is always element-wise. Use `np.dot()` for matrix multiplication.

**Q:** What does `a[::-1]` do?
**A:** Reverses the array by stepping backwards with step = -1.

**Q:** What does `arr[7:2:-1]` return for `arr = [10,20,30,40,50,60,70,80,90,100]`?
**A:** `[80 70 60 50 40]` — starts at index 7, walks backward, stops before index 2.

**Q:** What is the difference between `matrix[2, :]` and `matrix[2:3, :]`?
**A:** `matrix[2, :]` returns a 1D array shape `(4,)`; `matrix[2:3, :]` returns a 2D array shape `(1, 4)`.

**Q:** Why must you write `(arr > 5) & (arr < 20)` instead of `arr > 5 & arr < 20`?
**A:** `&` has higher precedence than `>`, so without parentheses NumPy evaluates `5 & arr` first, producing wrong results or an error.

**Q:** What does `arr[arr > 10] = 0` do?
**A:** Boolean mask assignment — sets every element greater than 10 to 0 in-place. This modifies the original array.

**Q:** What is the difference between `matrix[[0,2], [1,3]]` and `matrix[np.ix_([0,2], [1,3])]`?
**A:** `[[0,2],[1,3]]` pairs indices element-wise → returns `[matrix[0,1], matrix[2,3]]` (1D, 2 values). `np.ix_` selects a full 2×2 submatrix at rows 0,2 × cols 1,3.

**Q:** Does a basic slice return a view or a copy? How do you verify?
**A:** A view — it shares memory with the original. Verify with `result.base` — returns the source array (not `None`) if it's a view.


