print("Day 1 ready")
import math  # for math.sqrt [web:134]

def mean(xs):
    # TODO
    pass

def std(xs):
    # Population standard deviation: sqrt(mean((x - mu)^2))
    # TODO
    pass

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
