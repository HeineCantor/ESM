import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def lpq(block):
    block = np.reshape(block, (9, 9))
    transformedBlock = np.fft.fft2(block)
    transformedBlockShifted = np.fft.fftshift(transformedBlock)

    X1 = transformedBlockShifted[3, 4]
    X2 = transformedBlockShifted[3, 5]
    X3 = transformedBlockShifted[4, 5]
    X4 = transformedBlockShifted[5, 5]

    a = np.asarray([np.real(X1), np.imag(X1), np.real(X2), np.imag(X2), np.real(X3), np.imag(X3), np.real(X4), np.imag(X4)])
    c = a > 0

    return np.sum(c * 2 ** np.arange(8))

improntaImage = np.float64(io.imread('./Images/impronta100.png')) / 255.0
lpqImage = ndi.generic_filter(improntaImage, lpq, size=9)

hist, bins = np.histogram(lpqImage.flatten(), bins=np.arange(0, np.max(lpqImage)+2), density=True)

plt.subplot(1, 2, 1); plt.imshow(improntaImage, clim=None, cmap='gray'); plt.title('Original Image')
plt.subplot(1, 2, 2); plt.imshow(lpqImage, clim=None, cmap='gray'); plt.title('LPQ Image')

plt.figure()

plt.bar(np.arange(0, len(bins)-1), hist)

plt.show()