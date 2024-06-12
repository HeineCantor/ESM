import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.exposure import equalize_hist

def glob_equaliz(image):
    return equalize_hist(image) * 255

def blockEqualize(block):
    block = np.reshape(block, (3, 3))
    block = equalize_hist(block) * 255
    return block[1, 1]

def local_equaliz(image, windowSize):
    return ndi.generic_filter(image, blockEqualize, size=(windowSize)) * 255

originalImage = np.float64(io.imread('./Images/quadrato.tif')) / 255
globEqualizedImage = glob_equaliz(originalImage)
localEqualizedImage = local_equaliz(originalImage, 3)

plt.figure()

plt.subplot(1, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Quadrato")
plt.subplot(1, 3, 2); plt.imshow(globEqualizedImage, clim=None, cmap='gray'); plt.title("Global Equalized Quadrato")
plt.subplot(1, 3, 3); plt.imshow(localEqualizedImage, clim=None, cmap='gray'); plt.title("Local Equalized Quadrato")

plt.show()