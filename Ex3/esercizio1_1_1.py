import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

def smooth(image):
    MASK = np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        , dtype=np.float64
    ) / 16

    return ndi.correlate(image, MASK)

originalDorian = np.float64(io.imread('./Images/dorian.jpg')) / 255
smoothedDorian = smooth(originalDorian)

plt.subplot(1, 2, 1); plt.imshow(originalDorian, clim=None, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(smoothedDorian, clim=None, cmap='gray'); plt.title('Smoothed')

plt.show()