### Dot Product
* `np.dot(a,b)` → scalar for 1D arrays
* `a @ b` equivalent  
* `(a * b).sum()` manual version

### Cosine Similarity  
* Range **[-1, 1]**
* 1.0 = identical direction
* 0.0 = orthogonal
* `eps=1e-10` → avoid div-by-zero

### Matrix Multiplication
* Shape: `A(m×k) @ B(k×n) → C(m×n)`
* Inner dims **must match**
* Methods: `matmul/@/dot`

### Neural Net Forward Pass
* `X(batch,in) @ W(in,out) + b(out) → y(batch,out)`
* **W.shape = (input_features, output_neurons)**

