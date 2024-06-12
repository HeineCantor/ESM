import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import bitop

originalImage = io.imread('./Images/frattale.jpg')

imageBitplanes = [bitop.bitget(originalImage, i) for i in range(8)]

plt.figure()

plt.imshow(originalImage, clim=None, cmap='gray')

plt.figure()

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(imageBitplanes[i], clim=None, cmap='gray')
    plt.title("Bitplane " + str(i))

plt.show()