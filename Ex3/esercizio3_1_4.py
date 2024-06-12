import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

MASK_SIZE = 9

mask3 = np.triu(np.ones((MASK_SIZE, MASK_SIZE)), k=-1) - np.triu(np.ones((MASK_SIZE, MASK_SIZE)), k=2)
mask1 = mask3[:,::-1] # prendi tutte le righe di mask3, ma invertile

mask2 = np.zeros((MASK_SIZE, MASK_SIZE))
mask2[MASK_SIZE//2-1: MASK_SIZE//2+1] = 1    # tutte le righe da 3 a 6 saranno messe a 1
mask4 = mask2.T

maskList = [mask1, mask2, mask3, mask4]

def directionalVariance(input):
    actualMask = None
    minVariance = np.infty

    input = input.reshape((MASK_SIZE, MASK_SIZE))

    for mask in maskList:
        tempMasking = input * mask
        variance = np.var(tempMasking)
        if(variance < minVariance):
            minVariance = variance
            actualMask = tempMasking

    return np.mean(actualMask)

originalImage = np.fromfile('./Images/zebre.y', np.uint8)
originalImage = np.float64(np.reshape(originalImage, (321, 481)))

s = 25
additiveNoise = s*np.random.randn(321, 481)
noisyImage = originalImage + additiveNoise

plt.subplot(1,3,1); plt.imshow(noisyImage, clim=None, cmap='gray'); plt.title("Original Image")

filterImage = ndi.generic_filter(noisyImage, directionalVariance, (MASK_SIZE, MASK_SIZE))

plt.subplot(1,3,2); plt.imshow(filterImage, clim=None, cmap='gray'); plt.title("Directional Variance Filtering")

meanFiltered = ndi.uniform_filter(noisyImage, size=5)

plt.subplot(1,3,3); plt.imshow(meanFiltered, clim=None, cmap='gray'); plt.title("Uniform (Mean) Filtering")

mseDirVar = np.mean((filterImage-originalImage)**2)
psnrDirVar = 10*np.log10(255**2/mseDirVar)

mseMean = np.mean((meanFiltered-originalImage)**2)
psnrMean = 10*np.log10(255**2/mseMean)

print(f"PSNR Dir. Var. Filter: {psnrDirVar}")
print(f"PSNR Mean Filter: {psnrMean}")

plt.show()