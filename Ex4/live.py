import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from mpl_toolkits.mplot3d import Axes3D

originalImage = np.float64(io.imread('./Images/rettangolo.jpg'))
fftImage = np.fft.fft2(originalImage)

adjustedFFT = np.log(1+np.abs(np.fft.fftshift(fftImage)))

plt.figure()
plt.imshow(adjustedFFT, clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

M, N = originalImage.shape
P = 2*M
Q = 2*N

fftImage2 = np.fft.fft2(originalImage, (P, Q))

adjustedFFT2 = np.log(1+np.abs(np.fft.fftshift(fftImage2)))

m = np.fft.fftshift(np.fft.fftfreq(adjustedFFT2.shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(adjustedFFT2.shape[1]))

#ax3d = Axes3D(plt.figure())
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

l, k = np.meshgrid(n, m)

ax3d.plot_surface(l, k, adjustedFFT2, linewidth=0, cmap='jet')

plt.show()