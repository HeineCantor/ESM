import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def rgb2cmy(image):
    cmyImage = 1 - image
    return cmyImage

def rgb2cmyk(image):
    cmyImage = rgb2cmy(image)
    kImage = np.min(cmyImage, axis=2)
    cmykImage = np.dstack((cmyImage, kImage))
    return cmykImage

plt.close('all')

originalImage = np.float64(io.imread('./Images/fragole.jpg'))
originalImage /= 255

cmykImage = rgb2cmyk(originalImage)

plt.subplot(1, 5, 1); plt.imshow(originalImage); plt.title("RGB Image")
plt.subplot(1, 5, 2); plt.imshow(cmykImage[:,:,0], clim=None, cmap='gray'); plt.title("Cyan Component")
plt.subplot(1, 5, 3); plt.imshow(cmykImage[:,:,1], clim=None, cmap='gray'); plt.title("Magenta Component")
plt.subplot(1, 5, 4); plt.imshow(cmykImage[:,:,2], clim=None, cmap='gray'); plt.title("Yellow Component")
plt.subplot(1, 5, 5); plt.imshow(cmykImage[:,:,3], clim=None, cmap='gray'); plt.title("Black Component")

plt.show()