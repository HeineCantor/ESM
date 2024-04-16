import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

K = 15

originalImage = np.float64(io.imread('./Images/fragole.jpg'))
originalImage /= 255

filtered = ndi.uniform_filter(originalImage, (K, K, 1))

plt.subplot(1, 2, 1); plt.imshow(originalImage); plt.title("Original Image")
plt.subplot(1, 2, 2); plt.imshow(filtered); plt.title("Filtered Image")

plt.show()