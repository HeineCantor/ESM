import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

spazioImage = io.imread('./Images/space.jpg')
smoothedSpazio = ndi.uniform_filter(spazioImage, size=15)

THRESH = 0.25
smoothMax = np.max(smoothedSpazio)

thresholdedSpazio = np.zeros_like(smoothedSpazio)
mask = smoothedSpazio > THRESH * smoothMax

thresholdedSpazio = spazioImage * mask

plt.subplot(1, 2, 1); plt.imshow(spazioImage, clim=None, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(thresholdedSpazio, clim=None, cmap='gray'); plt.title('Smoothed')

plt.show()