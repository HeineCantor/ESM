import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from skimage.color import rgb2hsv, hsv2rgb

fotoOriginaleImage = np.float64(io.imread('./Images/foto_originale.tif')) / 255.0
hsvImage = rgb2hsv(fotoOriginaleImage)

intensity = hsvImage[:,:,2]

fotoTrasformata = np.fft.fft2(intensity)
fotoTrasformataShifted = np.fft.fftshift(fotoTrasformata)

M, N = intensity.shape

m, n = np.fft.fftshift(np.fft.fftfreq(M)), np.fft.fftshift(np.fft.fftfreq(N))

mu, ni = np.meshgrid(n, m)

H = (np.abs(mu) <= 0.10) & (np.abs(ni) <= 0.25)

intensitaTrasformataFiltrata = fotoTrasformataShifted * H
intensitaTrasformata = np.real(np.fft.ifft2(np.fft.ifftshift(intensitaTrasformataFiltrata)))

hsvImage[:,:,2] = intensitaTrasformata
fotoTrasformata = hsv2rgb(hsvImage)

plt.subplot(1, 3, 1); plt.imshow(fotoOriginaleImage)
plt.subplot(1, 3, 2); plt.imshow(np.log(1 + np.abs(fotoTrasformataShifted)), clim=None, cmap='gray')
plt.subplot(1, 3, 3); plt.imshow(fotoTrasformata, cmap='gray')

plt.show()