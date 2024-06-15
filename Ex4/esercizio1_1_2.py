import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

originalImage = np.float64(io.imread('./Images/volto.tif'))
transformedImage = np.fft.fftshift(np.fft.fft2(originalImage))

magnitude = np.abs(transformedImage)
enhancedMagnitude = np.log(1 + magnitude)

phase = np.angle(transformedImage)

plt.subplot(2, 3, 1); plt.imshow(originalImage, clim=None, cmap='gray'); plt.title('Original Image')
plt.subplot(2, 3, 2); plt.imshow(enhancedMagnitude, clim=None, cmap='gray'); plt.title('Enhanced Magnitude')
plt.subplot(2, 3, 3); plt.imshow(phase, clim=None, cmap='gray'); plt.title('Phase')

reconstructedMagnitude = np.fft.ifft2(np.fft.ifftshift(magnitude))
reconstructedPhase = np.fft.ifft2(np.fft.ifftshift(np.exp(1j * phase)))

plt.subplot(2, 3, 4); plt.imshow(np.abs(reconstructedMagnitude), clim=None, cmap='gray'); plt.title('Reconstructed Magnitude')
plt.subplot(2, 3, 5); plt.imshow(np.abs(reconstructedPhase), clim=None, cmap='gray'); plt.title('Reconstructed Phase')

plt.figure()

rettangoloImage = np.float64(io.imread('./Images/rettangolo.jpg'))
transformedRettangolo = np.fft.fftshift(np.fft.fft2(rettangoloImage))

magnitudeRettangolo = np.abs(transformedRettangolo)
phaseRettangolo = np.angle(transformedRettangolo)

reconstructedPhaseRettangolo = np.fft.ifft2(np.fft.ifftshift(magnitude*np.exp(1j * phaseRettangolo)))
reconstructedMagnitudeRettangolo = np.fft.ifft2(np.fft.ifftshift(magnitudeRettangolo*np.exp(1j * phase)))

plt.subplot(1, 2, 1); plt.imshow(np.abs(reconstructedMagnitudeRettangolo), clim=None, cmap='gray'); plt.title('Rectangle Magnitude + Phase')
plt.subplot(1, 2, 2); plt.imshow(np.abs(reconstructedPhaseRettangolo), clim=None, cmap='gray'); plt.title('Magnitude + Rectangle Phase')

plt.show()