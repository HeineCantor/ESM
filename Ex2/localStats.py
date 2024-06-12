import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

k0, k1, k2 = 0.4, 0.02, 0.4
E = 4

originalImage = np.float64(io.imread('./Images/filamento.jpg'))

mean, dev = np.mean(originalImage), np.std(originalImage)

meanImage = ndi.uniform_filter(originalImage, size=3)
devImage = ndi.generic_filter(originalImage, np.std, size=3)

mask = (((meanImage <= k0 * mean) & (devImage >= k1 * dev)) & (devImage <= k2 * dev))

enhancedImage = np.copy(originalImage)
enhancedImage[mask] *= E

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Filamento")
plt.subplot(1, 2, 2); plt.imshow(enhancedImage, clim=None, cmap='gray'); plt.title("Enhanced Filamento")

plt.show()