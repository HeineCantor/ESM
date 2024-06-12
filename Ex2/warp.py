import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.transform import warp

lenaImage = np.float64(io.imread('./Images/lena.jpg'))
lenaImage = lenaImage[252:277, 240:290] # occhio di lena

M, N = lenaImage.shape

# voglio ingrandire l'occhio di lena al doppio delle dimensioni
A = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64)

# interpolazione nearest neighbor
lenaNN = warp(lenaImage, A, order=0, output_shape=(2*M, 2*N))

# interpolazione bilineare
lenaBilinear = warp(lenaImage, A, order=1, output_shape=(2*M, 2*N))

# interpolazione bicubica
lenaBicubic = warp(lenaImage, A, order=3, output_shape=(2*M, 2*N))

plt.figure()

plt.subplot(1, 4, 1); plt.imshow(lenaImage, clim=None, cmap='gray')
plt.subplot(1, 4, 2); plt.imshow(lenaNN, clim=None, cmap='gray')
plt.subplot(1, 4, 3); plt.imshow(lenaBilinear, clim=None, cmap='gray')
plt.subplot(1, 4, 4); plt.imshow(lenaBicubic, clim=None, cmap='gray')

plt.show()