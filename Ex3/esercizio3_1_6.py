import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

vMask = np.array(
    [
        [ 0,  0,  0],
        [-1,  1,  0],
        [ 0,  0,  0],
    ]
)

hMask = np.array(
    [
        [ 0, -1,  0],
        [ 0,  1,  0],
        [ 0,  0,  0],
    ]
)

d1Mask = np.array(
    [
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  0],
    ]
)

d2Mask = np.array(
    [
        [ 0,  0,  0],
        [ 0,  1,  0],
        [-1,  0,  0],
    ]
)

originalImage = np.float64(io.imread('./Images/tetto.png'))

plt.subplot(1, 2, 1); plt.imshow(originalImage, clim=None, cmap='gray')

vImage = ndi.correlate(originalImage, vMask)
hImage = ndi.correlate(originalImage, hMask)
d1Image = ndi.correlate(originalImage, d1Mask)
d2Image = ndi.correlate(originalImage, d2Mask)

Qv = ndi.uniform_filter(vImage**2, 5)
Qh = ndi.uniform_filter(hImage**2, 5)
Qd1 = ndi.uniform_filter(d1Image**2, 5)
Qd2 = ndi.uniform_filter(d2Image**2, 5)

# Ugly af
#Qmin = np.minimum(np.minimum(Qv, Qh), np.minimum(Qd1, Qd2))

Qstack = np.stack([Qv, Qh, Qd1, Qd2], -1)
Qmin = np.min(Qstack, axis=-1)

maxQ = ndi.generic_filter(Qmin, np.max, (3, 3))

keypointMask = (maxQ > 20) & (maxQ == Qmin) # 500 è altino

plt.subplot(1, 2, 2); plt.imshow(keypointMask, clim=None, cmap='gray')

plt.show()