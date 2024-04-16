import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def rgb2cmy(image):
    cmyImage = 1 - image
    return cmyImage

originalImage = np.float64(io.imread('./Images/foto.jpg'))
originalImage /= 255

originalImage2 = np.float64(io.imread('./Images/foto_originale.tif'))
originalImage2 /= 255

gamma = [0.1, 0, 0]

cmyImage = rgb2cmy(originalImage)
cmyImage -= gamma
enhanched = rgb2cmy(cmyImage)

plt.subplot(1, 3, 1); plt.imshow(originalImage)
plt.subplot(1, 3, 2); plt.imshow(originalImage2)
plt.subplot(1, 3, 3); plt.imshow(enhanched)

plt.show()