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

# ── TASK 2: exp + log ──────────────────────────────────────────

print("\n── exp & log ──")
x = np.array([0.0, 1.0, 2.0, 3.0])

print("x:          ", x)
print("np.exp(x):  ", np.exp(x))          # e^0=1, e^1=2.718, e^2=7.389...
print("np.log(x+1):", np.log(x + 1))      # +1 avoids log(0) = -inf

# ── Softmax (the real ML formula) ──
scores = np.array([2.0, 1.0, 0.5])        # raw model outputs (logits)
exp_scores = np.exp(scores)
softmax = exp_scores / exp_scores.sum()   # divide by total so probs sum to 1

# first - make an array of the raw outputs
# second - get the exp value of teh array
# then divide the array by the total sum, it will convert it into a prob
# goal is to convert into prob.


print("\n── Softmax ──")
print("Scores:     ", scores)
print("exp(scores):", np.round(exp_scores, 4))   #np.round(array, decimals=0)
print("Softmax:    ", np.round(softmax, 4))
print("Sum:        ", softmax.sum())       # should be exactly 1.0

# ── Log-loss (binary cross-entropy for one sample) ──
# Log-loss (also called binary cross-entropy) measures 
# how far your model's predicted probability is from the actual true label.
y_true = 1                                 # actual label: positive class
y_pred = 0.9                               # model's predicted probability

loss = -( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )
# formula
#L=−[y⋅log(y^)+(1−y)⋅log(1−y^)]
print("\n── Log-loss ──")
print(f"y_true={y_true}, y_pred={y_pred}")
print(f"Loss (confident + correct):  {loss:.4f}")   # small loss

y_pred_bad = 0.05                          # model is confidently WRONG
loss_bad = -( y_true * np.log(y_pred_bad) + (1 - y_true) * np.log(1 - y_pred_bad) )
print(f"y_true={y_true}, y_pred={y_pred_bad}")
print(f"Loss (confident + wrong):    {loss_bad:.4f}")   # large loss

# bigger the value, bigger the loss


# ── TASK 3: Trig + np.where + np.clip ─────────────────────────

import numpy as np  # already imported, just for reference

# ── Trig ──
print("\n── Trig ──")
angles = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])  # 0°,30°,45°,90°,180°(in rad)

print("angles (rad):  ", np.round(angles, 4))
print("np.sin:        ", np.round(np.sin(angles), 4))
print("np.cos:        ", np.round(np.cos(angles), 4))

# ── np.where ──
print("\n── np.where ──")
x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])

# Syntax: np.where(condition, value_if_true, value_if_false)
relu = np.where(x > 0, x, 0.0)           # ReLU activation: pass positives, zero out negatives
print("x:             ", x)
print("ReLU(x):       ", relu)

# Label remap: mark positives as 1, non-positives as 0
labels = np.where(x > 0, 1, 0)
print("Labels (x>0):  ", labels)

# ── np.clip ──
print("\n── np.clip ──")
probs = np.array([0.0, 0.0001, 0.4, 0.9, 1.0])   # raw predicted probabilities

# Problem: log(0) = -inf, log(1) crashes log-loss
# Fix: clip to [1e-7, 1-1e-7] before computing log
clipped = np.clip(probs, 1e-7, 1 - 1e-7)  # array, min, max 
# np.clip() limits values in an array to a specified range — 
# anything below the minimum gets set to the minimum, 
# anything above the maximum gets set to the maximum.

print("raw probs:     ", probs)
print("clipped:       ", clipped)
print("log(clipped):  ", np.round(np.log(clipped), 4))   # no -inf!

# Gradient clipping example (prevents exploding gradients)
gradients = np.array([-50.0, -2.0, 0.5, 3.0, 100.0])
clipped_grads = np.clip(gradients, -5.0, 5.0)
print("\nraw grads:     ", gradients)
print("clipped grads: ", clipped_grads)

# ── TASK 4: Aggregation + axis control ────────────────────────

print("\n── Aggregation (1D) ──")
scores = np.array([0.8, 0.1, 0.05, 0.05])   # softmax output for 4 classes

print("scores:      ", scores)
print("sum:         ", np.sum(scores))        # should be 1.0
print("mean:        ", np.mean(scores))
print("std:         ", np.round(np.std(scores), 4))

# argmax and argmin produces the index of the max and min
print("argmax:      ", np.argmax(scores))     # index of predicted class → 0
print("argmin:      ", np.argmin(scores))     # index of least likely class

# ── 2D: batch of 3 samples, 4 class scores each ──
print("\n── Aggregation (2D) ──")
batch = np.array([
    [2.0, 1.0, 0.5, 0.1],   # sample 0
    [0.3, 3.5, 0.2, 0.8],   # sample 1
    [1.1, 0.9, 2.8, 0.4],   # sample 2
])
print("batch shape:", batch.shape)            # (3, 4)

print("\naxis=None (global):")
print("  mean:      ", np.mean(batch))        # single number, produces the mean of all the values

print("\naxis=0 (per column — across samples):")
# axis = 0 per column
print("  mean:      ", np.round(np.mean(batch, axis=0), 4))   # shape (4,) 


# axis = 1 per row
print("\naxis=1 (per row — across classes):")
print("  mean:      ", np.round(np.mean(batch, axis=1), 4))   # shape (3,)

# ── argmax per sample → predicted class for each ──
print("\nPredicted class per sample:")
print("  argmax:    ", np.argmax(batch, axis=1))   # should be [0, 1, 2]
