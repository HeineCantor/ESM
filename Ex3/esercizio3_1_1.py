import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.util import random_noise

SIZE_LIST = [5, 7, 9]
sizeListLen = len(SIZE_LIST)

plt.close('all')

originalImage = (io.imread('./Images/dorian.jpg'))
randomNoiseImage = 255*random_noise(originalImage)

originalImage = np.float64(originalImage)

plt.subplot(2,sizeListLen,1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Dorian")
plt.subplot(2,sizeListLen,2); plt.imshow(randomNoiseImage, clim=None, cmap='gray'); plt.title("Random Noise Dorian")

for i in range(sizeListLen):
    meadianFiltered = ndi.generic_filter(randomNoiseImage, np.median, (SIZE_LIST[i], SIZE_LIST[i]))
    plt.subplot(2, sizeListLen, 4+i); plt.imshow(meadianFiltered, clim=None, cmap='gray'); plt.title(f"Median Dorian ({SIZE_LIST[i]}x{SIZE_LIST[i]})")

    mse = np.mean((meadianFiltered-originalImage)**2)
    print(f"MSE ({SIZE_LIST[i]}x{SIZE_LIST[i]}): {mse}")

plt.show()