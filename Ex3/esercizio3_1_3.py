import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

TEST_STEPS = [10, 50, 100]

def enhanc(input, mask, epochs, a=0.073, b=0.177):
    hMask = np.array(
        [
            [a, b, a],
            [b, 0, b],
            [a, b, a]
        ]
    )

    initStep = input * mask

    processingStep = initStep

    for i in range(epochs):
        tempStep = ndi.correlate(processingStep, hMask)
        nextStep = processingStep * mask + tempStep * (1-mask)
        processingStep = nextStep

    return processingStep

plt.close('all')

originalImage = np.float64(io.imread('./Images/bebe.jpg'))
mask = np.float64(io.imread('./Images/mask.bmp'))

plt.subplot(1, len(TEST_STEPS)+1, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title("Original Image")

for i in range(len(TEST_STEPS)):
    enhanchedImage = enhanc(originalImage, mask, TEST_STEPS[i])
    plt.subplot(1, len(TEST_STEPS)+1, i+2); plt.imshow(enhanchedImage, clim=None, cmap='gray'); plt.title(f"Enhanced Image (K={TEST_STEPS[i]})")

plt.show()