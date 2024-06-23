import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def dehaze(image):
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    A1 = A2 = 0.7020
    A3 = 0.7098

    t0 = 0.1

    K = np.min(np.stack((red / A1, green / A2, blue / A3), 2), axis=2)
    xdark = ndi.generic_filter(K, np.min, size=15)

    t = 1 - 0.95 * xdark

    newRed = (red - A1) / np.maximum(t, t0) + A1
    newGreen = (green - A2) / np.maximum(t, t0) + A2
    newBlue = (blue - A3) / np.maximum(t, t0) + A3

    return np.dstack((newRed, newGreen, newBlue))

paesaggioImage = np.float64(io.imread('./Images/paesaggio.jpg')) / 255.0

paesaggioDehazed = dehaze(paesaggioImage)

plt.subplot(1, 2, 1); plt.imshow(paesaggioImage); plt.title('Original Image')
plt.subplot(1, 2, 2); plt.imshow(paesaggioDehazed); plt.title('Dehazed Image')

plt.show()