import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from skimage.filters import correlate_sparse

EPSILON = 2**(-60)
B = 10

def filtraggioGuidato(mask, guide, B):
    mask = mask / np.linalg.norm(mask)
    guide = guide / np.linalg.norm(guide)

    medMask = ndi.generic_filter(mask, np.mean, (B, B))
    medGuide = ndi.generic_filter(guide, np.mean, (B, B))

    varGuide = ndi.generic_filter(guide, np.var, (B, B))

    corr = ndi.uniform_filter(mask * guide, B)

    a = (corr - medGuide*medMask)/(varGuide+EPSILON)
    b = medMask - a*medGuide

    muA = ndi.generic_filter(a, np.mean, (B, B))
    muB = ndi.generic_filter(b, np.mean, (B, B))

    y = muA*guide+muB
    return y

originalMask = np.float64(io.imread('./Images/mask.png'))
guida = np.float64(io.imread('./Images/guida.png'))

result = filtraggioGuidato(originalMask, guida, B=B)

plt.subplot(2, 2, 1); plt.imshow(guida, clim=None, cmap='gray')
plt.subplot(2, 2, 2); plt.imshow(result, clim=None, cmap='gray')

plt.subplot(2, 2, 3); plt.imshow(guida*result, clim=None, cmap='gray')
plt.subplot(2, 2, 4); plt.imshow(originalMask*result, clim=None, cmap='gray')

plt.show()