import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

originalImage = np.float64(io.imread('./Images/volto.tif'))
plt.subplot(2, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray')

fft = np.fft.fftshift(np.fft.fft2(originalImage))
ampiezza = np.log(1+np.abs(fft))

plt.subplot(2, 3, 2); plt.imshow(ampiezza, clim=None, cmap='gray')

fase = np.angle(fft)

plt.subplot(2, 3, 3); plt.imshow(fase, clim=None, cmap='gray')

fftWithoutPhase = fft/np.abs(fft)
inverseNoPhase = np.real(np.fft.ifft2(fftWithoutPhase))

plt.subplot(2, 3, 4); plt.imshow(inverseNoPhase, clim=None, cmap='gray')

plt.show()