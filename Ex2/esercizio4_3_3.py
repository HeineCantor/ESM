import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.transform import warp
from skimage.data import checkerboard

def rot_shear(image, theta, c):
    M, N = image.shape

    T1 = np.array([[1, 0, 0], [0, 1, 0], [M/2, N/2, 1]], dtype=np.float64)
    T2 = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float64)
    T3 = np.array([[1, 0, 0], [c, 1, 0], [0, 0, 1]], dtype=np.float64)
    T4 = np.array([[1, 0, 0], [0, 1, 0], [-M/2, -N/2, 1]], dtype=np.float64)

    T = T4 @ T3 @ T2 @ T1
    A = T[[1, 0, 2],:][:, [1, 0, 2]].T

    return warp(image, A, order=1, output_shape=(M, N))

originalImage = np.float64(checkerboard())
rotated = rot_shear(originalImage, np.pi/4, 0.8)

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(originalImage, clim=None, cmap='gray')
plt.subplot(1, 2, 2); plt.imshow(rotated, clim=None, cmap='gray')

plt.show()