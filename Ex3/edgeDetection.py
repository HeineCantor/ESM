import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

mask1 = np.array(
    [
        [ -1,  -1,  -1],
        [  2,   2,   2],
        [ -1,  -1,  -1]
    ]
)

mask2 = np.array(
    [
        [ -1,  -1,   2],
        [ -1,   2,  -1],
        [  2,  -1,  -1]
    ]
)

mask3 = np.array(
    [
        [ -1,   2,  -1],
        [ -1,   2,  -1],
        [ -1,   2,  -1]
    ]
)

mask4 = np.array(
    [
        [  2,  -1,  -1],
        [ -1,   2,  -1],
        [ -1,  -1,   2]
    ]
)

plt.close('all')

figure, (ax0, ax1) = plt.subplots(1, 2)

originalImage = np.float64(io.imread("Images/quadrato.jpg"))
ax0.imshow(originalImage, clim=None, cmap='gray')

masked1 = ndi.correlate(originalImage, mask1)
masked2 = ndi.correlate(originalImage, mask2)
masked3 = ndi.correlate(originalImage, mask3)
masked4 = ndi.correlate(originalImage, mask4)

maskStruct = np.stack((masked1, masked2, masked3, masked4), -1)
maxImage = np.max(maskStruct, -1)

maxImage *= maxImage > 0.2*np.max(maxImage)

ax1.imshow(maxImage, clim=None, cmap='gray')

plt.show()