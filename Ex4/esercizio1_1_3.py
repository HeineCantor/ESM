import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

FFT_RESOLUTION = (256, 256)
k_list = [5, 10, 15]

filtriMedia = []
filtriTrasformati = []

for k in k_list:
    filtriMedia.append(np.ones((k, k)) / (k ** 2))

for i, filtro in enumerate(filtriMedia):
    trasformato = np.fft.fftshift(np.fft.fft2(filtro, FFT_RESOLUTION))
    moduloTrasformatoEnhanced = np.log(1+np.abs(trasformato))

    plt.subplot(1, len(filtriMedia), i+1); plt.imshow(moduloTrasformatoEnhanced, cmap='gray'); plt.title('Enhanced Fourier Transform\nk = ' + str(k_list[i]))

plt.show()