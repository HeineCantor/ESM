import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.transform import warp

def centralRotation(image, theta):
    M, N = image.shape
    
    T1 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [N/2, M/2, 1]
        ]
        , dtype=np.float64
    )

    T2 = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )

    T3 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [-N/2, -M/2, 1]
        ]
        , dtype=np.float64
    )

    T = T3 @ T2 @ T1
    A = T[[1, 0, 2],:][:, [1, 0, 2]].T
    
    return warp(image, A, order=0, output_shape=(M, N))

lenaImage = np.float64(io.imread('./Images/lena.jpg'))

rotated = centralRotation(lenaImage, np.pi/4)

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(lenaImage, clim=None, cmap='gray')
plt.subplot(1, 2, 2); plt.imshow(rotated, clim=None, cmap='gray')

plt.show()