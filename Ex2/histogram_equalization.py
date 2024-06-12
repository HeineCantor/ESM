import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.exposure import equalize_hist

def manual_equalize_hist(image):
    hist, bins = np.histogram(image, np.arange(257))
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)

    intImage = np.uint8(image)

    equalizedImage = cdf[intImage]

    return equalizedImage

originalImage = np.float64(io.imread('./Images/marte.jpg'))
#originalImage = np.float64(io.imread('./Images/quadrato.tif'))
hist, bins = np.histogram(originalImage, np.arange(257))

autoEqualizedImage = equalize_hist(originalImage) * 255
autoHist, autoBins = np.histogram(autoEqualizedImage, np.arange(257))

manualEqualizedImage = manual_equalize_hist(originalImage) * 255
manualHist, manualBins = np.histogram(manualEqualizedImage, np.arange(257))

plt.bar(np.arange(256), hist)
plt.bar(np.arange(256), autoHist)
plt.bar(np.arange(256), manualHist)

plt.figure()

plt.subplot(1, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Marte")
plt.subplot(1, 3, 2); plt.imshow(autoEqualizedImage, clim=None, cmap='gray'); plt.title("Auto Equalized Marte")
plt.subplot(1, 3, 3); plt.imshow(manualEqualizedImage, clim=None, cmap='gray'); plt.title("Manual Equalized Marte")

plt.show()