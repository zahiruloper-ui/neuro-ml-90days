import numpy as np

print("=== Task 1: Random seed demo ===")

# Without seed (will vary each run)
print("No seed:")
a1 = np.random.rand(3)
print("rand(3):", a1)

# With seed (always same)
print("\nWith seed=42:")
np.random.seed(42)   # picks from the 42nd random sequence of numbers
a2 = np.random.rand(3)
print("rand(3):", a2)

# Reset seed and verify reproducibility
print("\nReset seed=42 (same as above):")
np.random.seed(42)
a3 = np.random.rand(3)
print("rand(3):", a3)
print("a2 == a3?", np.array_equal(a2, a3))

# Different seed gives different sequence
print("\nSeed=123:")
np.random.seed(123)
a4 = np.random.rand(3)
print("rand(3):", a4)


print("\n=== Task 2: Different random distributions ===")

np.random.seed(42)  # Reset for consistency

# Integers: randint(low, high, size) - [low, high)
print("randint(1, 7, 5):   ", np.random.randint(1, 7, 5))  # Dice rolls #(lowest value possible, highest value possible, size )

# Uniform [0,1)
print("rand(5):           ", np.random.rand(5))

# Uniform custom range
print("uniform(2, 5, 5):  ", np.random.uniform(2, 5, 5))  # [2, 5)

# Normal/Gaussian (mean=0, std=1 by default)
print("normal(0,1,5):     ", np.random.normal(0, 1, 5))

# Normal with mean=100, std=15 (like IQ scores)
print("normal(100,15,5):  ", np.random.normal(100, 15, 5))

print("\nLarge sample normal (check bell shape):")
big_sample = np.random.normal(0, 1, 1000)
print(f"Mean: {big_sample.mean():.3f}, Std: {big_sample.std():.3f}")


print("\n=== Task 3: Array-wise random operations ===")

np.random.seed(42)  # Reset sequence

# 2D random array
print("rand(3,4):")
arr_2d = np.random.rand(3, 4)   # (rows, cols)
print(arr_2d)
print(f"Shape: {arr_2d.shape}")

# choice() - sample from existing array
colors = np.array(['red', 'green', 'blue'])
print("\nchoice from colors (with replacement):")
samples = np.random.choice(colors, size=5, replace=True)
print(samples)

# shuffle() - in-place permutation
numbers = np.arange(10)
print("\nBefore shuffle:", numbers)
np.random.shuffle(numbers)
print("After shuffle: ", numbers)

# permutation() - returns shuffled copy
print("\npermutation([1,2,3]):", np.random.permutation([1, 2, 3]))  # same thing as shuffle 
                                                                    # only difference returns a whole new array


# Your code structure
np.random.seed(42)  # Already set from Task 3

# 1. Fake dataset: features + labels (100 samples)
n_samples = 100
X = np.random.randn(n_samples, 4) * 2 + 3  # 4 features, mean~3, std~2 # randn is normal(bell curve)
y = np.random.randint(0, 2, n_samples)     # Binary labels
print("Fake dataset shape:", X.shape, y.shape)
print("X mean/std:", X.mean(axis=0).round(2), X.std(axis=0).round(2))

# 2. Weight init (tiny net: 3→5→1 neurons)
W1 = np.random.randn(3, 5) * 0.1
W2 = np.random.randn(5, 1) * 0.1
print("\nW1 shape:", W1.shape, "mean:", W1.mean().round(3))

# 3. Data augmentation example (shuffle + noise)
orig_data = np.arange(10).reshape(2,5)
print("\nOriginal:", orig_data)

# Shuffle rows
shuffled_rows = np.random.permutation(orig_data)
print("Shuffled rows:", shuffled_rows)

# Add noise
noisy = orig_data + np.random.randn(2,5) * 0.1
print("Noisy:", noisy.round(2))
