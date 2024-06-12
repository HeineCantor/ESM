import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

NOISE_DEV = 20

dorianImage = np.float64(io.imread('./Images/dorian.jpg'))
M, N = dorianImage.shape

noise = np.random.randn(M, N) * NOISE_DEV
noisyDorian = dorianImage + noise

for FILTER_SIZE in [3, 5, 7, 9, 11]:
    cleansedDorian = ndi.uniform_filter(noisyDorian, size=FILTER_SIZE)

    mse = np.mean((dorianImage - cleansedDorian) ** 2)
    print(f"MSE for filter size {FILTER_SIZE}: {mse}")

    plt.subplot(1, 3, 1); plt.imshow(dorianImage, clim=None, cmap='gray'); plt.title('Original')
    plt.subplot(1, 3, 2); plt.imshow(noisyDorian, clim=None, cmap='gray'); plt.title('Noisy')
    plt.subplot(1, 3, 3); plt.imshow(cleansedDorian, clim=None, cmap='gray'); plt.title('Cleansed')

    plt.show()

