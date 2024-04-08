import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

x = np.float64(io.imread('Images/circuito_rumoroso.jpg'))
y = ndi.generic_filter(x, np.median, (5,5))
z = ndi.generic_filter(y, np.min, (5,5))
plt.subplot(1,3,1); plt.imshow(x,clim=[0,255],cmap='gray')
plt.subplot(1,3,2); plt.imshow(y,clim=[0,255],cmap='gray')
plt.subplot(1,3,3); plt.imshow(z,clim=[0,255],cmap='gray')

plt.show()