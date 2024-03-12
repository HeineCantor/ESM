import numpy as np

TEST_ARRAY = [-10, 3, -6, 0, 1, -2, 3, 4, -15, 3, 21]
LIMIT = 8

def clip(x, limit):
    return [min(value, limit) for value in x]

array = np.asarray(TEST_ARRAY)
clipped = clip(array, LIMIT)

print(clipped)