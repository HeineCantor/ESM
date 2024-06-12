import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.transform import warp

def centralRotation(image, theta):
    M, N = image.shape
    
    A1 = np.array(
        [
            [1, 0, N/2],
            [0, 1, M/2],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )

    A2 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )

    A3 = np.array(
        [
            [1, 0, -N/2],
            [0, 1, -M/2],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )

    A = A1 @ A2 @ A3
    
    return warp(image, A, order=0, output_shape=(M, N))

lenaImage = np.float64(io.imread('./Images/lena.jpg'))

rotated = centralRotation(lenaImage, np.pi/4)

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(lenaImage, clim=None, cmap='gray')
plt.subplot(1, 2, 2); plt.imshow(rotated, clim=None, cmap='gray')

plt.show()