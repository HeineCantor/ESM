import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def thresholding_locale(x):
    a = 30; b = 1.5

    devImage = ndi.generic_filter(x, np.std, size=3)
    mask = (x > a * devImage) & (x > b * np.mean(x))

    return mask

yeastImage = np.float64(io.imread('./Images/yeast.tif')) / 255.0
mask = thresholding_locale(yeastImage)

plt.subplot(1, 2, 1); plt.imshow(yeastImage, cmap='gray'); plt.title('Original Image')
plt.subplot(1, 2, 2); plt.imshow(mask, cmap='gray'); plt.title('Mask')

plt.show()