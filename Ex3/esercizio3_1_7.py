import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

m1 = np.array(
    [
        [  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0],
        [  1,  1,  1,  1,  1,  1,  1],
        [  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0],
    ]
)

antim1 = np.ones((7, 7)) - m1

m2 = np.array(
    [
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
    ]
)

m3 = np.array(
    [
        [  1,  0,  0,  0,  0,  0,  0],
        [  0,  1,  0,  0,  0,  0,  0],
        [  0,  0,  1,  0,  0,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  0,  0,  1,  0,  0],
        [  0,  0,  0,  0,  0,  1,  0],
        [  0,  0,  0,  0,  0,  0,  1],
    ]
)

m4 = np.array(
    [
        [  0,  0,  0,  0,  0,  0,  1],
        [  0,  0,  0,  0,  0,  1,  0],
        [  0,  0,  0,  0,  1,  0,  0],
        [  0,  0,  0,  1,  0,  0,  0],
        [  0,  0,  1,  0,  0,  0,  0],
        [  0,  1,  0,  0,  0,  0,  0],
        [  1,  0,  0,  0,  0,  0,  0],
    ]
)

def meanDifferenceM1(input):
    reshaped = np.reshape(input, (7, 7))
    whiteMean = np.mean(reshaped * m1)
    blackMean = np.mean(reshaped * antim1)

    return whiteMean-blackMean

x = io.imread('./Images/retina.tif')

R = x[:,:,0]
G = x[:,:,1]
B = x[:,:,2]

filteredG = ndi.generic_filter(G, meanDifferenceM1, (7, 7))

totalImage = np.stack([R, filteredG, B], -1)
plt.subplot(2,2,1); plt.imshow(totalImage)

plt.show()