import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

originalImage = io.imread('./Images/granelli.jpg')
hist, bins = np.histogram(originalImage, np.arange(257))

plt.figure()

plt.bar(np.arange(256), hist)

plt.show()

imgMin = np.min(originalImage)
imgMax = np.max(originalImage)

fshsImage = (originalImage - imgMin) / (imgMax - imgMin) * 255

histFSHS, binsFSHS = np.histogram(fshsImage, np.arange(257))

plt.figure()

plt.bar(np.arange(256), histFSHS)

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(originalImage, clim=[0, 255], cmap='gray'); plt.title("Original Granelli")
plt.subplot(1, 2, 2); plt.imshow(fshsImage, clim=[0, 255], cmap='gray'); plt.title("FSHS Granelli")

plt.show()