import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import bitop

originalImage = np.fromfile('./Images/lena.y', dtype=np.uint8)
originalImage = np.reshape(originalImage, (512, 512))

marchio = np.fromfile('./Images/marchio.y', dtype=np.uint8)
marchio = np.reshape(marchio, (350, 350))

# Pad the watermark with zeros
marchio = np.pad(marchio, ((81, 81), (81, 81)), 'constant', constant_values=(0, 0))

watermarked = bitop.bitset(originalImage, 0, marchio)

plt.subplot(1, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray')
plt.subplot(1, 3, 2); plt.imshow(marchio, clim=None, cmap='gray')
plt.subplot(1, 3, 3); plt.imshow(watermarked, clim=None, cmap='gray')

plt.show()