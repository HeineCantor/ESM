from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

img1 = np.float32(io.imread('./Images/img1.jpg')) / 255.0
img2 = np.float32(io.imread('./Images/img2.jpg')) / 255.0
img3 = np.float32(io.imread('./Images/img3.jpg')) / 255.0

lbp1 = local_binary_pattern(img1, 8, 1, 'uniform')
lbp2 = local_binary_pattern(img2, 8, 1, 'uniform')
lbp3 = local_binary_pattern(img3, 8, 1, 'uniform')

hist1, bins2 = np.histogram(lbp1.flatten(), bins=np.arange(0, np.max(lbp1) + 2), density=True)
hist2, bins2 = np.histogram(lbp2.flatten(), bins=np.arange(0, np.max(lbp2) + 2), density=True)
hist3, bins2 = np.histogram(lbp3.flatten(), bins=np.arange(0, np.max(lbp3) + 2), density=True) 

plt.figure()

plt.subplot(1, 3, 1); plt.bar(np.arange(10), hist1); plt.title('Histogram 1')
plt.subplot(1, 3, 2); plt.bar(np.arange(10), hist2); plt.title('Histogram 2')
plt.subplot(1, 3, 3); plt.bar(np.arange(10), hist3); plt.title('Histogram 3')

plt.show()