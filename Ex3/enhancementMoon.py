import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

hMask = np.array(
    [
        [  0,  -1,   0],
        [ -1,   5,  -1],
        [  0,  -1,   0]
    ]
)

plt.close('all')

figure, (ax0, ax1) = plt.subplots(1, 2)

originalImage = np.float64(io.imread("Images/luna.jpg"))
ax0.imshow(originalImage, clim=None, cmap='gray')

detection = ndi.correlate(originalImage, hMask)
ax1.imshow(detection, clim=None, cmap='gray')

plt.show()