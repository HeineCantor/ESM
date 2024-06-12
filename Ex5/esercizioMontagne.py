import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

plt.close('all')

originalImage = np.float64(io.imread('./Images/montagna.jpg'))
originalImage /= 255

enhanced = originalImage**0.8

plt.subplot(1, 2, 1); plt.imshow(originalImage); plt.title("Original Image")
plt.subplot(1, 2, 2); plt.imshow(enhanced); plt.title("Enhanced Image")

plt.show()