import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

xAxis = [i*5 for i  in range(1, 11)]
yAxisAdaptive = []
yAxisUniform = []

GAUSSIAN_VARIANCE_STEP = 5

def adaptiveFilter(inputImage):
    localMean = np.mean(inputImage)
    localVariance = np.var(inputImage)

    middlePixel = inputImage[inputImage.shape[0]//2]
    return middlePixel - (gaussianVariance/localVariance)*(middlePixel-localMean)

gaussianVariance = GAUSSIAN_VARIANCE_STEP

originalImage = np.float64(io.imread("Images/barbara.gif")[0,:,:])
M, N = originalImage.shape

for i in range(1, 11):
    gaussianVariance = i*GAUSSIAN_VARIANCE_STEP
    additiveNoise = gaussianVariance*np.random.randn(M, N)
    noisyImage = originalImage + additiveNoise

    #fig, (ax0, ax1) = plt.subplots(1, 2)
    #ax0.imshow(noisyImage, clim=None, cmap='gray')

    filteredImage = ndi.generic_filter(originalImage, adaptiveFilter, size=(7, 7))
    smoothedImage = ndi.uniform_filter(noisyImage, size=3)

    #ax1.imshow(filteredImage, clim=None, cmap='gray')

    mseAdaptive = np.mean((filteredImage-originalImage)**2)
    mseUniform = np.mean((smoothedImage-originalImage)**2)

    yAxisAdaptive.append(mseAdaptive)
    yAxisUniform.append(mseUniform)

    print(f"Noisy Sigma: {gaussianVariance} | Adaptive MSE: {mseAdaptive} - Uniform MSE: {mseUniform}")

plt.plot(xAxis, yAxisAdaptive, label="Adaptive Filter")
plt.plot(xAxis, yAxisUniform, label="Uniform Filter")

plt.legend()

plt.show()