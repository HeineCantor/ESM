import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import bitop

originalImage = io.imread('./Images/frattale.jpg')

#imageBitplanes = [bitop.bitget(originalImage, i) for i in range(8)]
reconstructions = []

for i in range(8):
    reconstructions.append(np.copy(originalImage))
    for j in range(i+1):
        reconstructions[i] = bitop.bitset(reconstructions[i], j, 0)

plt.figure()

for i, reconstruction in enumerate(reconstructions):
    plt.subplot(2, 4, i + 1)
    plt.imshow(reconstruction, clim=None, cmap='gray')
    plt.title("Reconstruction " + str(i))

plt.show()