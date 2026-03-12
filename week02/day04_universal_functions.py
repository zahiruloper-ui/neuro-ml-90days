import numpy as np
import time    # Used to measure how long code takes to ru

arr = np.array([-4.0, 0.0, 9.0, 16.0, 25.0])

# --- 1. sqrt: square root element-wise ---
print("arr:         ", arr)
print("np.sqrt:     ", np.sqrt(np.abs(arr)))   # abs first so no NaN on negatives

# --- 2. abs: absolute value element-wise ---
print("np.abs:      ", np.abs(arr))

# --- 3. square: x^2 element-wise ---
print("np.square:   ", np.square(arr))

# --- 4. Speed comparison: ufunc vs Python loop ---
big = np.arange(1_000_000, dtype=float)

# Python loop
start = time.time()
loop_result = [x**0.5 for x in big]
loop_time = time.time() - start

# Ufunc
start = time.time()
ufunc_result = np.sqrt(big)
ufunc_time = time.time() - start

print(f"\nLoop time:  {loop_time:.4f}s")
print(f"Ufunc time: {ufunc_time:.4f}s")
print(f"Speedup:    {loop_time / ufunc_time:.1f}x faster")

