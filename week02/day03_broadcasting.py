# Task 1: Broadcasting rules 

import numpy as np

# --- Rule 1: Pad shorter shape with 1s on the LEFT ---
# Shape (3, 4) + shape (4,) → treated as (3,4) + (1,4) → result (3,4)
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])        # shape (4,)
result = A + b

print(f"\nA shape: {A.shape}  +  b shape: {b.shape}")
print(f"b is treated as shape (1, 4) → result shape: {result.shape}")

print(result) # [[2, 3, 4, 5]
              #  [2, 3, 4, 5]
              #  [2, 3, 4, 5]]


# --- Rule 2: Stretch size-1 dimensions ---
# Shape (3, 1) + shape (5,) → (3,1) + (1,5) → result (3,5)
col = np.array([[1], [2], [3]])    # shape (3, 1)
row = np.array([10, 20, 30, 40, 50])  # shape (5,) → treated as (1, 5)
result2 = col + row

print(f"\ncol shape: {col.shape}  +  row shape: {row.shape}")
print("Result shape (3,5):\n")
# in the resulting shape each col will be added 5 times to make 5 fols
# Moreover, to only 1 can be stretched
print(result2)  # [[11, 21, 31, 41, 51]
                #  [12, 22, 32, 42, 52]
                #  [13, 23, 33, 43, 53]] 


# --- Rule 3: Incompatible — triggers ValueError ---
# Shape (3, 4) + shape (3,) → (3,4) vs (1,3) → trailing dims 4 != 3 → Error
print("\nAttempting incompatible broadcast (3,4) + (3,):")
try:
    bad = np.ones((3, 4)) + np.ones((3,))
except ValueError as e:
    print(f"  ValueError caught: {e}")




print("\nTASK 2: Scalar + 1D + 2D Broadcasting\n")


M = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9,10,11,12]])   # shape (3, 4)

# --- Scalar broadcast ---
scalar_result = M * 2
# shape stays the same, just each element has to go through the cond.
print(f"\nM * 2  (scalar broadcast, shape stays {scalar_result.shape}):")
print(scalar_result)   # [[2,4,6,8],
                       # [10,12,14,16],
                       # [18,20,22,24]]
                       

# --- 1D row broadcast ---
row_vec = np.array([0, 1, 2, 3])  # shape (4,)
row_result = M + row_vec
print(f"\nM + [0,1,2,3]  (row broadcast, shape {row_result.shape}):")
print(row_result)    # [[1, 3, 5, 7],
                     #  [5, 7, 9, 11],
                     #  [9,11,13,15]]

# --- 1D col broadcast ---
col_vec = np.array([[100], [200], [300]])  # shape (3, 1)
col_result = M + col_vec
print(f"\nM + [[100],[200],[300]]  (col broadcast, shape {col_result.shape}):")
print(col_result) # [[101, 102, 103, 104],
                  # [205, 206, 207, 208],
                  # [309,310,311,312]]

# --- Row + Col → 2D outer-style ---
r = np.array([[0, 10, 20, 30]])   # shape (1, 4)
c = np.array([[0], [1], [2]])     # shape (3, 1)
outer_result = r + c
print(f"\nr shape {r.shape} + c shape {c.shape}  → both stretch:")
print(outer_result)
# [[0, 10, 20, 30],
# [1, 11, 21, 31],
# [2,12,22,32]]
print(f"Result shape: {outer_result.shape}")




print("TASK 3: Practical Broadcasting")


data = np.array([[1.0,  2.0,  3.0,  4.0],
                 [5.0,  6.0,  7.0,  8.0],
                 [9.0, 10.0, 11.0, 12.0]])   # shape (3, 4)

# --- Row normalization: subtract each ROW's mean ---
# keepdims=True preserves shape as (3,1) instead of collapsing to (3,)
row_means = data.mean(axis=1, keepdims=True)   # shape (3, 1)

row_normalized = data - row_means
print(f"\nrow_means shape: {row_means.shape}")  # (3,1)
print(f"row_means:\n{row_means}")   # [[2.5], [6.5], [10.5]]
print(f"Row-normalized (each row sums to ~0):\n{row_normalized}") 
#[[- 1.5,  - 0.5, 0,5,  1.5],
# [- 1.5,  - 0.5, 0,5,  1.5],
# [- 1.5,  - 0.5, 0,5,  1.5]]
print(f"Row sums (should be 0): {row_normalized.sum(axis=1)}")

# --- Col normalization: subtract each COLUMN's mean ---
col_means = data.mean(axis=0, keepdims=True)   # shape (1, 4)
col_normalized = data - col_means
print(f"\ncol_means shape: {col_means.shape}") # (1,4)
print(f"col_means:\n{col_means}")  # [[5.0, 6.0, 7.0, 8.0]]
print(f"Col-normalized (each col sums to ~0):\n{col_normalized}")
#[[-4., -4., -4.,-4.],
# [0.,  0.,  0.,  0.],
# [4., 4., 4.,4.]]
print(f"Col sums (should be 0): {col_normalized.sum(axis=0)}")

# --- Bias addition: add a bias vector to every row (neural net style) ---
bias = np.array([0.5, -0.5, 1.0, -1.0])       # shape (4,) → broadcast as (1,4)
output = data + bias
print(f"\nbias shape: {bias.shape}") # (4,)
print(f"data + bias (bias added to every row):\n{output}")
#[[1.5,  1.5,  4.,  3.],
#[5.5,  5.5,  8.,  7.],
#[9.5, 9.5, 12., 11.]]




print("TASK 4: Shape Mismatches + Fixes")


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])          # shape (3, 3)
v = np.array([10, 20, 30])        # shape (3,)

# --- Mismatch 1: trying to add col-wise but shape is wrong ---
print("\n[Mismatch 1] A + v where we WANT col-wise addition:")
print(f"  A.shape={A.shape}, v.shape={v.shape}") # (3,3) (3,)
print(f"  NumPy treats v as (1,3) → adds to every ROW (probably not what you want)")
print(f"  Result (row-wise, may be unintended):\n{A + v}")

# Fix: reshape v to (3,1) so it broadcasts down each column
v_col = v[:, np.newaxis]           # shape (3, 1)  ← newaxis fix
# : keeps all the elements in the row, np.newaxis creates a new axis (1)
print(f"\n  Fix: v[:, np.newaxis] → shape {v_col.shape}") # (3, 1)
print(f"  Result (col-wise, intended):\n{A + v_col}")  
#[[11, 12, 13],
# [24, 25, 26],
# [37, 38, 39]]



# Same fix with reshape
v_col2 = v.reshape(-1, 1)          # shape (3, 1)
# -1 indicates to figure it out according to v, 1 means number of columns are 1

print(f"\n  Same fix with .reshape(-1,1) → shape {v_col2.shape}")
print(f"  Same result? {np.array_equal(A + v_col, A + v_col2)}") 

# --- Mismatch 2: genuine ValueError — incompatible, needs diagnosis ---
x = np.ones((4, 3))               # shape (4, 3)
y = np.ones((4,))                 # shape (4,)

print("\n[Mismatch 2] x.shape=(4,3)  y.shape=(4,)")
print("  Trailing dims: 3 vs 4 → incompatible → ValueError")
try:
    _ = x + y
except ValueError as e:
    print(f"  Error: {e}")

# Fix: make y a column vector (4,1) so it broadcasts across columns
y_col = y.reshape(-1, 1)          # shape (4, 1)
fixed = x + y_col
print(f"\n  Fix: y.reshape(-1,1) → shape {y_col.shape}")
print(f"  Result shape: {fixed.shape}")
print(f"  Result:\n{fixed}")

# --- Quick reference: newaxis positions ---
z = np.array([1, 2, 3, 4])        # shape (4,)
print(f"\n[newaxis reference] z.shape = {z.shape}")
print(f"  z[np.newaxis, :]  → shape {z[np.newaxis, :].shape}  (row vector)")
print(f"  z[:, np.newaxis]  → shape {z[:, np.newaxis].shape}  (col vector)")
