import numpy as np

#1D Slicing


arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


# Basic start/stop/step

print(arr[2:7])        # [30, 40, 50, 60, 70]
print(arr[::2])        # [10, 30, 50, 70, 90]
print(arr[1::3])       # [20,50,80]

# Negative Indexing
print(arr[-1])         # 100  
print(arr[-3:])        # [80, 90, 100] , starts from -3
print(arr[:-2])        # [10, 20, 30, 40, 50, 60, 70, 80]


# Negative step (reversal)
print(arr[::-1])       # [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
print(arr[7:2:-1])     # [80,70,60, 50, 40]
print(arr[::-2])       # [100, 80, 60, 40, 20]

# Edge cases
print(arr[5:2])        # []
print(arr[5:100])      # [60, 70, 80, 90, 100] -out-of-bounds stop is safe, clips 
print(arr[-100:3])     # [10, 20, 30]  — out-of-bounds negative start also clips 

# TASK 2: 2D Slicing

matrix = np.array([
    [1,  2,  3,  4],   # row 0
    [5,  6,  7,  8],   # row 1
    [9,  10, 11, 12],  # row 2
    [13, 14, 15, 16]   # row 3
])

print("\nTask 2\n")

# Single element
print(matrix[1, 2])       # 7

# Full row / full column
print(matrix[2, :])       # [9, 10, 11, 12]
print(matrix[:, 1])       # [2, 6, 10, 14]

# Submatrix — rows 0–1, cols 1–2
print(matrix[0:2, 1:3])   # [[2,3]
                          #  [6,7]]

# Every other row, every other col
print(matrix[::2, ::2])   # [[1,3]
                          #  [9,11]]


# Negative indexing
print(matrix[-1, :])      # [13, 14, 15, 16]
print(matrix[:, -1])      # [4, 8, 12, 16]
print(matrix[-2:, -2:])   # [[11,12]
                          #  [15, 16]]
                          

# Reverse rows (flip vertically)
print(matrix[::-1, :])    # [[13, 14, 15, 16]
                          #  [9, 10, 11, 12]
                          #  [5, 6, 7, 8]
                          #  [1, 2, 3, 4]] 

# Reverse cols (flip horizontally)
print(matrix[:, ::-1])    # [[4, 3, 2, 1]
                          #  [8, 7, 6, 5]
                          #  [12, 11, 10, 9]
                          #  [16, 15, 14, 13]]

print("\nTask 3\n")

# TASK 3: Boolean Masks

arr = np.array([3, 15, 7, 42, 8, 23, 4, 16, 11, 30])

# ── Creating masks ──
mask_gt10 = arr > 10
print(mask_gt10)        # [False, True, False, True, True, True, True, True]
print(arr[mask_gt10])   # [15, 42, 23, 16, 11, 30]

# Inline (no named variable needed)
print(arr[arr % 2 == 0])   # [42, 8, 4, 16, 30]
print(arr[arr < 10])       # [3, 7, 8, 4]

# ── Combining conditions ──
# AND — both must be True (use & for and)
print(arr[(arr > 5) & (arr < 20)])  # [15, 7, 8, 16, 11]


# OR — at least one must be True (use | for or)
print(arr[(arr < 5) | (arr > 25)])  # [3, 42, 4, 30]

# NOT — invert a mask (use ~ for not condition)
print(arr[~(arr > 10)])           # [3, 7, 8, 4]      

# ── 2D boolean masking ──
matrix = np.array([
    [1,  2,  3],
    [4,  5,  6],
    [7,  8,  9]
])


# can use boolean mask [] for 2d as well

print(matrix[matrix > 4])         #[5, 6, 7, 8, 9]  converts into 1D
                                                   
matrix_copy = matrix.copy()
matrix_copy[matrix_copy % 2 == 0] = 0  #[[1, 0, 3]   # = some value, allocates the value to those cells 
                                       # [0,5,0]     # that satisfy the condition
                                       # [7,0,9]]
print(matrix_copy)

# ── np.where — conditional replacement ──
arr2 = np.array([10, 25, 3, 18, 7, 40])
result = np.where(arr2 > 15, arr2, -1)   # keep value if >15, else replace with -1 (cond., df, replaced value)
print(result)    #[-1, 25, -1, -1, 18, -1, 40]


print("\n Task 4 \n")

# ── TASK 4: Fancy Indexing ───────────────────────────────────────

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# ── Integer array indexing (1D) ──
idx = [0, 2, 5, 7]      # if df[another df] then tthe ant=other one acts as index
print(arr[idx])         # [10, 30, 60, 80]  

# Reorder / repeat elements freely
print(arr[[3, 3, 1, 6]])  # [40, 40, 20, 70]

# Negative indices work too
print(arr[[-1, -3, 0]])   # [80, 60, 10]

# ── Fancy indexing on 2D ──
matrix = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

# Select specific rows

#[[rows], [cols]]
print(matrix[[0, 2]])         #  [[1, 2, 3, 4]
                              #   [9, 10, 11, 12]] 
print(matrix[[3, 1, 0]])      #  [[13, 14, 15, 16]
                              #   [5, 6, 7, 8]
                              #   [1, 2, 3, 4]]    

# Select specific rows AND specific cols (paired — not a submatrix!)
# make pairs
rows = [0, 1, 2]
cols = [0, 1, 2]
print(matrix[rows, cols])   # [1, 6, 11]  

# To get a submatrix with fancy indexing, use np.ix_
print(matrix[np.ix_([0, 2], [1, 3])])   # [[2, 4]
                                        #  [10, 12]]


# ── View vs Copy ── (CRITICAL difference)
arr = np.array([1, 2, 3, 4, 5])

# Slicing → VIEW (shares memory)
# It will remember and change the orginal one as well
# if slicing(:) is used
view = arr[1:4] 
view[0] = 99
print(arr)       

# Fancy indexing → COPY (independent memory)
# will not change the original
arr = np.array([1, 2, 3, 4, 5])
copy = arr[[1, 2, 3]]
copy[0] = 99
print(arr)        

# Boolean masking → also a COPY
# boolean mask will also not change the original
arr = np.array([1, 2, 3, 4, 5])
b_copy = arr[arr > 2]
b_copy[0] = 99
print(arr)        

# Force a copy from a slice explicitly
# .copy() on a slicing will make it have the property of a copy
arr = np.array([1, 2, 3, 4, 5])
safe = arr[1:4].copy()
safe[0] = 99
print(arr)        
