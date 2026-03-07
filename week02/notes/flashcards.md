## Week 2 Flashcards

**Q49:** What is the difference between `np.arange` and `np.linspace`?
**A49:** `arange` takes a step size (stop excluded); `linspace` takes a point count (stop included).

**Q50:** What are the four key array attributes and what does each return?
**A50:** `.shape` (dimension tuple), `.ndim` (axis count), `.dtype` (element type), `.size` (total elements).

**Q51:** What is the default dtype of `np.zeros()` and `np.ones()`?
**A51:** `float64`.

**Q52:** What does `m[:, 2]` return on a 2D array?
**A52:** The entire column at index 2 — all rows, col 2.

**Q53:** Why does `a * b` NOT give matrix multiplication in NumPy?
**A53:** `*` is always element-wise. Use `np.dot()` for matrix multiplication.

**Q54:** What does `a[::-1]` do?
**A54:** Reverses the array by stepping backwards with step = -1.

