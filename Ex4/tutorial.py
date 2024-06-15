import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

rettangoloImage = np.float64(io.imread('./Images/rettangolo.jpg'))
M, N = rettangoloImage.shape
Q, P = 2*M, 2*N

X = np.fft.fft2(rettangoloImage, (Q, P))    # opzionale: per specificare i passi di campionamento (Q, P)

X_shifted = np.fft.fftshift(X)

plt.figure()

plt.subplot(2, 2, 1); plt.imshow(rettangoloImage, cmap='gray'); plt.title('Original Image')
plt.subplot(2, 2, 2); plt.imshow(np.abs(X), cmap='gray', extent=(0, 1, 1, 0)); plt.title('Fourier Transform')

plt.subplot(2, 2, 3); plt.imshow(np.log(1 + np.abs(X)), cmap='gray', extent=(0, 1, 1, 0)); plt.title('Enhanced Fourier Transform')
plt.subplot(2, 2, 4); plt.imshow(np.log(1 + np.abs(X_shifted)), cmap='gray', extent=(-0.5, +0.5, +0.5, -0.5)); plt.title('Enhanced Shifted Fourier Transform')

Y = np.log(1 + np.abs(X_shifted))

m = np.fft.fftshift(np.fft.fftfreq(Y.shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(Y.shape[1]))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

l, k = np.meshgrid(n, m)
ax3d.plot_surface(l, k, Y, cmap='jet')

plt.show()