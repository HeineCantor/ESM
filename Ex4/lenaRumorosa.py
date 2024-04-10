import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

originalImage = np.fromfile("./Images/lenarumorosa.y", np.int16)
originalImage = np.float64(np.reshape(originalImage, (512, 512)))

M, N = originalImage.shape

m = np.fft.fftshift(np.fft.fftfreq(M))
n = np.fft.fftshift(np.fft.fftfreq(N))

lenaFFT = np.fft.fftshift(np.fft.fft2(originalImage))
lenaFFTamplitude = np.log(1+np.abs(lenaFFT))

D0 = 0.03
nu = 0.2

l, k = np.meshgrid(n, m)
D1 = np.sqrt((k+nu)**2 + (l-nu)**2)
D2 = np.sqrt((k-nu)**2 + (l+nu)**2)

H = (D1 > D0) & (D2 > D0)

filtered = H * lenaFFT
invertedFiltered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))

plt.subplot(2, 2, 1); plt.imshow(originalImage, clim=None, cmap='gray')
plt.subplot(2, 2, 2); plt.imshow(lenaFFTamplitude, clim=None, cmap='gray')
plt.subplot(2, 2, 3); plt.imshow(H, clim=[0, 1], cmap='gray', extent=(-0.5, +0.5, +0.5, -0.5))
plt.subplot(2, 2, 4); plt.imshow(invertedFiltered, clim=None, cmap='gray')

plt.show()