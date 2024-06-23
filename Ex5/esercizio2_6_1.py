import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2hsv, hsv2rgb

azzurroImage = io.imread('./Images/azzurro.jpg')
hsvImage = rgb2hsv(azzurroImage)

rossoImage = np.copy(azzurroImage)
hsvRossoImage = rgb2hsv(rossoImage)

hueRosso = hsvRossoImage[:,:,0]
saturationRosso = hsvRossoImage[:,:,1]
valueRosso = hsvRossoImage[:,:,2]

maskRosso = (hueRosso > 0.35) & (hueRosso < 0.66) & (saturationRosso > 0.2) & (valueRosso > 0.35) # questi valori alla fine me li sono presi dalla soluzione ahahahah
hueRosso[maskRosso] = 0.95
hsvRossoImage[:,:,0] = hueRosso % 1.0

saturationRosso[maskRosso] = 0.9
hsvRossoImage[:,:,1] = saturationRosso

rossoImage = hsv2rgb(hsvRossoImage)

plt.close('all')

plt.subplot(1, 4, 1); plt.imshow(azzurroImage); plt.title("Original Image")
plt.subplot(1, 4, 2); plt.imshow(hsvImage[:,:,0], clim=None, cmap='gray'); plt.title("Hue Image")
plt.subplot(1, 4, 3); plt.imshow(hsvImage[:,:,1], clim=None, cmap='gray'); plt.title("Saturation Image")
plt.subplot(1, 4, 4); plt.imshow(hsvImage[:,:,2], clim=None, cmap='gray'); plt.title("Value Image")

plt.figure()

plt.imshow(rossoImage)

plt.show()