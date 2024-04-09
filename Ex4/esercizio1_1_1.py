import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

BASIC_PATH = "./Images/"
IMAGES_PATH = ["circuito.jpg", "impronta.jpg", "anelli.tif"]

for i, path in enumerate(IMAGES_PATH):
    originalImage = np.float64(io.imread(BASIC_PATH + path))
    plt.subplot(2, len(IMAGES_PATH), i+1); plt.imshow(originalImage, clim=None, cmap='gray')    

    fftImage = np.fft.fftshift(np.fft.fft2(originalImage))
    adjustedFFT = np.log(1+np.abs(fftImage))

    plt.subplot(2, len(IMAGES_PATH), i+1+len(IMAGES_PATH)); plt.imshow(adjustedFFT, clim=None, cmap='gray', extent=(-0.5, +0.5, +0.5, -0.5))

plt.show()