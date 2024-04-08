import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

hMask = np.array(
    [
        [ -1,  -1,  -1],
        [ -1,   8,  -1],
        [ -1,  -1,  -1]
    ]
)

plt.close('all')

figure, (ax0, ax1) = plt.subplots(1, 2)

originalImage = np.float64(io.imread("Images/turbina.jpg"))
ax0.imshow(originalImage, clim=None, cmap='gray')

detection = ndi.correlate(originalImage, hMask)
threshold = 0.10*(np.max(detection)-np.min(detection))+np.min(detection)

detection *= detection < threshold

ax1.imshow(detection, clim=None, cmap='gray')

plt.show()