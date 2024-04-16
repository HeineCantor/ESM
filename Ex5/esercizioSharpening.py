import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

enhancingMask2D = np.float64(np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]
))

originalImage = np.float64(io.imread('./Images/fiori.jpg'))
originalImage /= 255

enhancingMask3D = np.expand_dims(enhancingMask2D, axis=-1)

enhanced = ndi.correlate(originalImage, enhancingMask3D)

plt.subplot(1, 2, 1); plt.imshow(originalImage); plt.title("Original Image")
plt.subplot(1, 2, 2); plt.imshow(enhanced); plt.title("Enhanced Image")

plt.show()