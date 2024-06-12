
import numpy as np
from skimage.feature import local_binary_pattern

def getLBPHist(image, P=8, R=1, normalize=False):
    lbpImage = local_binary_pattern(image, P=P, R=R)
    hist, bins = np.histogram(lbpImage, np.arange(257), density=True)
    if(normalize):
        hist = hist / np.linalg.norm(hist)
    return hist