import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import sys
sys.path.insert(0, "./")
import esmLib as esm

plt.close('all')

originalImage = np.float64(io.imread('./Images/brick.png'))

plt.subplot(1, 2, 1); plt.imshow(originalImage, cmap='gray'); plt.title('Original Image')

hist = esm.getLBPHist(originalImage, P=8, R=1)
plt.subplot(1, 2, 2); plt.bar(np.arange(256), hist); plt.title('LBP Histogram')

plt.show()