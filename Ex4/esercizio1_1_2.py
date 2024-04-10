import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

originalImage = np.float64(io.imread('./Images/volto.tif'))
plt.subplot(2, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray')

fft = np.fft.fftshift(np.fft.fft2(originalImage))

ampiezza = np.abs(fft)
fase = np.angle(fft)

ampiezzaLog = np.log(1+ampiezza)

plt.subplot(2, 3, 2); plt.imshow(ampiezzaLog, clim=None, cmap='gray')

fase = np.angle(fft)

plt.subplot(2, 3, 3); plt.imshow(fase, clim=None, cmap='gray')

fftWithoutPhase = fft/np.e**(np.angle(fft))
inverseNoPhase = np.real(np.fft.ifft2(np.fft.ifftshift(fftWithoutPhase)))

fftWithoutAmplitude = fft/np.abs(fft)
inverseNoAmplitude = np.real(np.fft.ifft2(np.fft.ifftshift(fftWithoutAmplitude)))

plt.subplot(2, 3, 5); plt.imshow(inverseNoAmplitude, clim=None, cmap='gray')
plt.subplot(2, 3, 6); plt.imshow(inverseNoPhase, clim=None, cmap='gray')

rectangle = np.float64(io.imread('./Images/rettangolo.jpg'))

fftRettangolo = np.fft.fftshift(np.fft.fft2(rectangle))

ampiezzaRettangolo = np.abs(fftRettangolo)
faseRettangolo = np.angle(fftRettangolo)

ampiezzaRettangoloLog = np.log(1+ampiezzaRettangolo)

plt.subplot(2, 2, 1); plt.imshow(ampiezzaRettangoloLog, clim=None, cmap='gray', extent=(-0.5, +0.5))

plt.figure()

plt.show()