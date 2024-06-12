import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

originalImage = np.float64(io.imread('./Images/mammografia.jpg')) / 255
negativeImage = - originalImage + 255

plt.subplot(1, 2, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Mammografia")
plt.subplot(1, 2, 2); plt.imshow(negativeImage, clim=None, cmap='gray'); plt.title("Negative Mammografia")

plt.show()
