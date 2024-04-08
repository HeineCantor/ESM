import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

h1 = np.array(
    [
        [  0,   1,   2],
        [ -1,   0,   1],
        [ -2,  -1,   0]
    ]
)

h2 = np.array(
    [
        [ -2,  -1,   0],
        [ -1,   0,   1],
        [  0,   1,   2]
    ]
)

plt.close('all')

figure, (ax0, ax1) = plt.subplots(1, 2)

originalImage = np.float64(io.imread("Images/angiogramma.jpg"))
ax0.imshow(originalImage, clim=None, cmap='gray')

h1Filtered = ndi.correlate(originalImage, h1)
h2Filtered = ndi.correlate(originalImage, h2)

grad = np.sqrt(h1Filtered**2+h2Filtered**2)
threshold = 1.5*np.mean(grad)

grad *= grad > threshold

ax1.imshow(grad, clim=None, cmap='gray')

plt.show()