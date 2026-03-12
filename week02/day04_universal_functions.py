import numpy as np
import time    # Used to measure how long code takes to ru

arr = np.array([-4.0, 0.0, 9.0, 16.0, 25.0])

# --- 1. sqrt: square root element-wise ---
print("arr:         ", arr)
print("np.sqrt:     ", np.sqrt(np.abs(arr)))   # abs first so no NaN on negatives
#[2., 0., 3., 4., 5.]
# np.abs() produces absolute values, np. sqrt produces sqrt values
# --- 2. abs: absolute value element-wise ---

print("np.abs:      ", np.abs(arr)) #[4.0, 0.0, 9.0, 16.0, 25.0]


# --- 3. square: x^2 element-wise ---
print("np.square:   ", np.square(arr)) # [16.0, 0.0, 81.0, 256.0, 625.0]

# --- 4. Speed comparison: ufunc vs Python loop ---
big = np.arange(1_000_000, dtype=float)

# creates 1D array of  1 million floats

# Python loop
start = time.time()
# time.time() returns the current time as a number — specifically, 
# the number of seconds that have elapsed 
# since January 1, 1970 (called the "epoch").
loop_result = [x**0.5 for x in big]
loop_time = time.time() - start

# Ufunc
start = time.time()
ufunc_result = np.sqrt(big)
ufunc_time = time.time() - start

print(f"\nLoop time:  {loop_time:.4f}s")
print(f"Ufunc time: {ufunc_time:.4f}s")
if ufunc_time > 0:
    print(f"Speedup:    {loop_time / ufunc_time:.1f}x faster")
else:
    print(f"Speedup:    >1000x faster (ufunc too fast to measure!)")


