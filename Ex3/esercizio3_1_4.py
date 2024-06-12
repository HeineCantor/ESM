import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

MASK_SIZE = 9
NOISE_DEV = 25

def directionalFiltering(image):
    filteredImage = np.zeros_like(image)
    minimumLocalVar = np.Infinity
    for mask in [mask1, mask2, mask3, mask4]:
        tmpFiltered = ndi.convolve(image, mask)
        localVar = np.var(tmpFiltered)
        if localVar < minimumLocalVar:
            minimumLocalVar = localVar
            filteredImage = tmpFiltered

    return filteredImage

mask1 = np.triu(np.ones((MASK_SIZE, MASK_SIZE)), -1)[::-1, ::] - np.triu(np.ones((MASK_SIZE, MASK_SIZE)), 2)[::-1, ::]

mask2 = np.zeros((MASK_SIZE, MASK_SIZE))
mask2[3:6, :] = 1

mask3 = np.triu(np.ones((MASK_SIZE, MASK_SIZE)), -1) - np.triu(np.ones((MASK_SIZE, MASK_SIZE)), 2)

mask4 = np.zeros((MASK_SIZE, MASK_SIZE))
mask4[:, 3:6] = 1

zebreImage = np.fromfile('./Images/zebre.y', dtype=np.uint8)
zebreImage = np.float64(zebreImage.reshape(321, 481))

M, N = zebreImage.shape

noise = NOISE_DEV * np.random.randn(M, N)
noisyZebre = zebreImage + noise

filteredZebre = directionalFiltering(noisyZebre)

plt.figure()

plt.subplot(1, 3, 1); plt.imshow(zebreImage, clim=None, cmap='gray'); plt.title('Original Image')
plt.subplot(1, 3, 2); plt.imshow(noisyZebre, clim=None, cmap='gray'); plt.title('Noisy Image')
plt.subplot(1, 3, 3); plt.imshow(filteredZebre, clim=None, cmap='gray'); plt.title('Filtered Image')

plt.show()