print("Day 1 ready")
import math  # for math.sqrt [web:134]

def mean(xs):
    if len(xs) == 0:  # len returns number of items 
        raise ValueError("mean() requires atleast one value") # raise throws an error, 
                                                              # ValueError is a type of error when type is 
                                                             # okay but value is not okay
    return sum(xs)/len(xs) # sum returns the sum of all items

def std(xs):
    # Population standard deviation: sqrt(mean((x - mu)^2))
    if len(xs) == 0:  # len returns number of items 
        raise ValueError("std() requires atleast one value") 
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)

def minmax_scale(xs):
    # Scale to [0, 1]. If all values equal, return all 0.0
    # TODO
    pass

# ---- tests ----
tests = [
    [1, 2, 3, 4, 5],
    [10, 10, 10],
    [-2, 0, 2],
]

for xs in tests:
    print("xs =", xs)
    print("mean =", mean(xs))
    print("std  =", std(xs))
    print("mm   =", minmax_scale(xs))
    print()
