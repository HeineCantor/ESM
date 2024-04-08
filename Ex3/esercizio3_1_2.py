import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

laplacianMask = np.array(
    [
        [  0,  -1,   0],
        [ -1,   5,  -1],
        [  0,  -1,   0]
    ]
)

laplacianMask2 = np.array(
    [
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [-1,  0,  5,  0, -1],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
    ]
)

plt.close('all')

originalImage = np.float64(io.imread('./Images/luna.jpg'))
laplacianImage = ndi.correlate(originalImage, laplacianMask)
laplacianImage2 = ndi.correlate(originalImage, laplacianMask2)

plt.subplot(1,3,1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Image")
plt.subplot(1,3,2); plt.imshow(laplacianImage, clim=None, cmap='gray'); plt.title("Classical Laplacian Image")
plt.subplot(1,3,3); plt.imshow(laplacianImage2, clim=None, cmap='gray'); plt.title("New Laplacian Image")

plt.show()