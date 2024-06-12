import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.transform import warp

def deforma(x, c, d):
    M, N = x.shape

    T1 = np.array([[1, 0, 0], [c, 1, 0], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, d, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    T = T1 @ T2
    A = T[[1, 0, 2],:][:, [1, 0, 2]].T
    return warp(x, A, order=3, output_shape=(M, N))

lenaImage = np.float64(io.imread('./Images/lena.jpg'))

deformata = deforma(lenaImage, 0.1, -0.4)

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(lenaImage, clim=None, cmap='gray')
plt.subplot(1, 2, 2); plt.imshow(deformata, clim=None, cmap='gray')

plt.show()