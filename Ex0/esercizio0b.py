import numpy as np

CLIP_COUNT = 4

def clip(x):
    flattened = x.flatten(order='C')
    flattened.sort()
    return flattened[-CLIP_COUNT:]

#matrix = np.asarray(TEST_MATRIX)
matrix = np.random.rand(16, 16)
clippedMatrix = clip(matrix)
print(f"MATRIX: \n {matrix}")
print(clippedMatrix)

