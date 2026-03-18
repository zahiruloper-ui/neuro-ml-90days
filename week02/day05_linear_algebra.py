import numpy as np

print("=== 1D dot products ===")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Three equivalent ways
dot1 = np.dot(a, b)   # (1*4)+(2*5)+(3*6)        # function form [web:1][web:3][web:8]
dot2 = a.dot(b)              # method form [web:3]
dot3 = (a * b).sum()         # manual: elementwise * then sum [web:5]

print("a:", a)
print("b:", b)
print("np.dot(a, b):", dot1)
print("a.dot(b):", dot2)
print("(a * b).sum():", dot3)



print("\n=== Cosine similarity (1D vectors) ===")

def cosine_similarity(u, v, eps=1e-10):
    u = np.asarray(u, dtype=float)    # np.asarray() converts inputs into array
    v = np.asarray(v, dtype=float)

    dot = np.dot(u, v)                    # numerator 
    norm_u = np.linalg.norm(u)           # magnitude of u
    norm_v = np.linalg.norm(v)           # magnitude of v 

    if norm_u < eps or norm_v < eps:     # avoid division by zero 
        return 0.0

    return dot / (norm_u * norm_v)       # value between -1 and 1 

u = np.array([1, 0])   #np.linalg.norm() works liek this : Sqrt(a^2 + b^2 + ......)
v = np.array([0, 1])
w = np.array([1, 1])

print("cos(u, u) (same direction):", cosine_similarity(u, u))      # should be 1.0
print("cos(u, v) (orthogonal):", cosine_similarity(u, v))          # ~0.0
print("cos(u, w):", cosine_similarity(u, w))                       # ~0.707

print("\n=== Task 2: Matrix multiplication ===")

# 2x3 matrix A, 3x2 matrix B → 2x2 result   (a of first x b of second) # cond (b of first = a of second)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print("A (2x3):")  
print(A)
print("B (3x2):")
print(B)

# Three equivalent ways for matrices
matmul1 = np.matmul(A, B)    # Preferred: matmul (short for math multiplication)
matmul2 = A @ B              # Python 3.5+: @ operator 
matmul3 = np.dot(A, B)       # Also works for 2D (matrix mult) 

print("\nA @ B (2x2 result):")
print("np.matmul(A, B):\n", matmul1) # [[27,64]
                                     #  [139, 154]]
print("A @ B:\n", matmul2)
print("np.dot(A, B):\n", matmul3)

# Verify all three are identical
print("\nAll three identical?", np.allclose(matmul1, matmul2) and np.allclose(matmul2, matmul3))


print("\n=== Task 3: Transpose (.T) ===")

# Original 2x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("A (2x3):")
print(A)
print("A.shape:", A.shape) #(2,3)

# Transpose: rows ↔ columns
A_T = A.T  # or A.transpose()
A_T_1 = A.transpose()
print("\nA.T (3x2):")
print(A_T) #[[1,4]
           # [2,5]
           # [3,6]]
print("A.T.shape:", A_T.shape) # (3,2)

# Verify: (A @ B) needs matching inner dimensions
print("\n=== Shape fixing demo ===")
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3x2

print("A (2x3) @ B (3x2): OK → 2x2")
print("A.shape:", A.shape, "B.shape:", B.shape)

print("\n=== Task 4: Linear Layer Forward Pass (X @ W + b) ===")

# Neural net linear layer: y = X @ W + b
# Batch of 3 samples, 4 features each → 3 outputs # rows are the samples 
                                                  # and features are the cols
X = np.array([[1, 2, 3, 4],    # Sample 1
              [5, 6, 7, 8],    # Sample 2  
              [9, 10, 11, 12]]) # Sample 3
print("X (batch, features):", X.shape)  # (3, 4)

W = np.array([[0.1, 0.2, 0.3],   # Weights (each col is the weight of each sample) 
              [0.4, 0.5, 0.6],   # good for mult
              [0.7, 0.8, 0.9],
              [1.0, 1.1, 1.2]])   # (4, 3)
print("W (features, outputs):", W.shape)  # (4, 3)

b = np.array([0.1, 0.2, 0.3])     # Bias (outputs,)
print("b (outputs,):", b.shape)    # (3,)

# Forward pass! Core of every neural net layer
y = X @ W + b

print("\nForward pass: y = X @ W + b")
print("X (3,4) @ W (4,3) + b (3,) → y (3,3)")
print("y.shape:", y.shape)
print("y:")
print(y)

# Verify first element manually:
# Sample 1: [1,2,3,4] @ first column of W + b[0]
# 1*0.1 + 2*0.4 + 3*0.7 + 4*1.0 + 0.1 = 0.1 + 0.8 + 2.1 + 4.0 + 0.1 = 7.1
print("\nManual check [0,0]:", y[0,0])
