import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

immagineLena = np.float64(io.imread('./Images/lena.jpg'))
X = np.fft.fftshift(np.fft.fft2(immagineLena))
M, N = X.shape

m = np.fft.fftshift(np.fft.fftfreq(M))
n = np.fft.fftshift(np.fft.fftfreq(N))

l, k = np.meshgrid(n, m)

D = np.sqrt(l**2 + k**2)
D0 = 0.1
D1 = 0.11

H = -np.exp(-D**2 / (2 * D0**2)) + np.exp(-D**2 / (2 * D1**2))

plt.figure()

plt.subplot(1, 2, 1); plt.imshow(H, cmap='gray', extent=(-0.5, +0.5, +0.5, -0.5)); plt.title('Filter H')

plt.show()
