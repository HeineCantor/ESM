import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def meanDifference(input, mask):
    input = input.reshape((MASK_SIZE, MASK_SIZE))

    whiteInput = input*mask
    blackInput = input*(1-mask)

    return np.mean(whiteInput) - np.mean(blackInput)

MASK_SIZE = 7

mask1 = np.zeros((MASK_SIZE, MASK_SIZE))
mask1[MASK_SIZE//2] = 1

mask2 = mask1.T

mask3 = np.diag(np.ones(MASK_SIZE), k=0)
mask4 = mask3[:,::-1]

x = io.imread('./Images/retina.tif')

R = x[:,:,0]
G = x[:,:,1]
B = x[:,:,2]

stackedImage = np.stack([R, G, B], -1)

plt.subplot(1, 2, 1); plt.imshow(stackedImage)

maskImage1 = ndi.generic_filter(G, meanDifference, (MASK_SIZE, MASK_SIZE), extra_arguments=tuple([mask1]))
maskImage2 = ndi.generic_filter(G, meanDifference, (MASK_SIZE, MASK_SIZE), extra_arguments=tuple([mask2]))
maskImage3 = ndi.generic_filter(G, meanDifference, (MASK_SIZE, MASK_SIZE), extra_arguments=tuple([mask3]))
maskImage4 = ndi.generic_filter(G, meanDifference, (MASK_SIZE, MASK_SIZE), extra_arguments=tuple([mask4]))

imageStack = np.stack([maskImage1, maskImage2, maskImage3, maskImage4], -1)
minImage = np.min(imageStack, axis=-1)

thresholdMap = minImage > -5
minImage *= thresholdMap

plt.subplot(1, 2, 2); plt.imshow(minImage, clim=None, cmap='gray')
plt.show()