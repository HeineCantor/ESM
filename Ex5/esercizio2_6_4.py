import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

demosaicMask = np.array(
    [
        [-1, 0, 1],
        [ 0, 1, 0],
        [ 1, 0, -1],
    ]
)

plt.close('all')

def demosa1(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]

    pass

originalImage = np.float64(io.imread('./Images/Fiori_mosaic.bmp'))

stacked = np.stack((originalImage[:,:,0], originalImage[:,:,1], originalImage[:,:,2]), axis=-1)
stacked /= 255

plt.imshow(stacked)

demosaicMask = np.expand_dims(demosaicMask, axis=-1)
demosaicked = ndi.correlate(stacked, demosaicMask)

plt.imshow(demosaicked)

# plt.subplot(1, 4, 1); plt.imshow(stacked); plt.title("Original Image")
# plt.subplot(1, 4, 2); plt.imshow(originalImage[:,:,0]); plt.title("Original Image")
# plt.subplot(1, 4, 3); plt.imshow(originalImage); plt.title("Original Image")
# plt.subplot(1, 4, 4); plt.imshow(originalImage); plt.title("Original Image")

plt.show()