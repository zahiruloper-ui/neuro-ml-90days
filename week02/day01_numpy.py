import numpy as np

#  use of np.array

a = np.array([1,2,3,4,5])         # 1D from a list
b = np.array([[1,2,3], [4,5,6]])  # 2D from a nested list

print("np.array 1D:", a)
print("np.array 2D:", b)


# use of np.zeros / np.ones

z = np.zeros((2,3))   # (rows,cols)
o = np.ones((3,2))    # (rows, cols)

# creates float by default
print("np.zeros\n", z)  # creates only 0s
print("np.ones\n", o)   # creates only 1s

# use of np.arange


r1 = np.arange(10)
r2 = np.arange(0, 20, 2)    #(start, end(excluded), add)
r3 = np.arange(1.0, 2.0, 0.25) 
print("\nnp.arange(10):", r1)
print("np.arange(0,20,2):", r2)
print("np.arange float:", r3)

# use of np.linspace

l1 = np.linspace(0,1,5)    # divide 0-1(inclusive) in 5 even pieces
l2 = np.linspace(0,10,11)


# float by default
print("\nnp.linspace(0,1,5):", l1)
print("np.linspace(0,10,11):", l2)


# Array attributes
a1 = np.array([10, 20, 30, 40])
a2 = np.array([[1, 2, 3],
               [4, 5, 6]])
a3 = np.zeros((2,3,4)) # 3D  # (separate table no., rows, cols)

# use of shape (dimension size)
print("a1.shape:", a1.shape)     # (4,)
print("a2.shape:", a2.shape)     # (2, 3)
print("a3.shape:", a3.shape)     # (2, 3, 4)

# use of .ndim (dimension no.)
print("a1.ndim:", a1.ndim)     #  1
print("a2.ndim:", a2.ndim)     # 2
print("a3.ndim:", a3.ndim)     # 3


# use of .dtype (data type of elements)
print("a1.dtype:", a1.dtype)     #  int32 or int64 (OS-dependent)
print("a2.dtype:", a2.dtype)     
print("a3.dtype:", a3.dtype)     

f = np.array([1.0, 2.0])
print("float array dtype:", f.dtype)   # float64

# use of .size, total number of elements

print("a1.size:", a1.size)     
print("a2.size:", a2.size)     
print("a3.size:", a3.size)   

# Bonus: force a dtype at creation
i32 = np.array([1, 2, 3], dtype=np.int32)
f32 = np.array([1, 2, 3], dtype=np.float32)
print("\nint32 array:", i32.dtype)
print("float32 array:", f32.dtype)



# Vectorized arithmetics

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Array + Array (element-wise)
print("a + b:", a + b)
print("b - a:", b - a)
print("b * a:", b * a)
print("b / a:", b / a)
print("a ** 2", a ** 2)

#Array + Scalar (broadcasts scalar to every element)

print("\na + 100:", a + 100)
print("a * 2.5:", a * 2.5)
print("a / 2:", a / 2)


# 2D array arithmetic

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])


print("\nm1 + m2:\n", m1 + m2)
print("m1 * m2 (element-wise):\n", m1 * m2)   # NOT matrix mult!
print("m1 ** 2:\n", m1 ** 2)


# Useful aggregate ops

c = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("\nSum:", c.sum())
print("Mean:", c.mean())
print("Max:", c.max())
print("Min:", c.min())
print("Std:", c.std().round(4))