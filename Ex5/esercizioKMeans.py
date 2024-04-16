import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from sklearn.cluster import k_means

clusterNum = 3

originalImage = np.float64(io.imread('./Images/fiori.jpg'))
originalImage /= 255

d = np.reshape(originalImage, (-1, 3))
centroid, idx, sum_var = k_means(d, clusterNum)
y = np.reshape(idx, originalImage.shape[:-1])

plt.imshow(y)
plt.show()