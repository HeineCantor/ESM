import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def fshs(image, K=255):
    imageMin = np.min(image)
    imageMax = np.max(image)

    return (image - imageMin) / (imageMax - imageMin) * K

spettroImage = np.float64(io.imread('./Images/spettro.jpg'))
logImage = fshs(np.log(spettroImage + 1))

vistaAereaImage = np.float64(io.imread('./Images/vista_aerea.jpg'))
powerVistaAerea = fshs(vistaAereaImage ** 3)

spinaDorsaleImage = np.float64(io.imread('./Images/spina_dorsale.jpg'))
powerSpinaDorsale = fshs(spinaDorsaleImage ** 0.3)

plt.figure()

plt.subplot(3, 2, 1); plt.imshow(spettroImage, clim=None, cmap='gray'); plt.title("Original Spettro")
plt.subplot(3, 2, 2); plt.imshow(logImage, clim=None, cmap='gray'); plt.title("Log Spettro")

plt.subplot(3, 2, 3); plt.imshow(vistaAereaImage, clim=None, cmap='gray'); plt.title("Original Vista Aerea")
plt.subplot(3, 2, 4); plt.imshow(powerVistaAerea, clim=None, cmap='gray'); plt.title("Power Vista Aerea")

plt.subplot(3, 2, 5); plt.imshow(spinaDorsaleImage, clim=None, cmap='gray'); plt.title("Original Spina Dorsale")
plt.subplot(3, 2, 6); plt.imshow(powerSpinaDorsale, clim=None, cmap='gray'); plt.title("Power Spina Dorsale")

plt.show()