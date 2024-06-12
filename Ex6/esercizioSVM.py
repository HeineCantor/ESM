import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, "./")
import esmLib as esm

TRAIN_DATA_PATH = './Images/train_lbp_8_1_default.npy'
TRAIN_LABEL_PATH = './Images/train_label.npy'

EPSILON = 1e-15

plt.close('all')

originalImage = np.float64(io.imread('./Images/brick.png'))

xTraining = np.load(TRAIN_DATA_PATH)
yTraining = np.load(TRAIN_LABEL_PATH)

# Feature normalization for training

xMean = np.mean(xTraining, axis=0)
xStd = np.std(xTraining, axis=0)

xTraining = (xTraining - xMean) / (xStd + EPSILON) # safety value e to avoid division by zero

classifier = LinearSVC(max_iter=100000, dual=True).fit(xTraining, yTraining)

testImage = np.float64(io.imread('./Images/test_breakhis.png'))
hist = esm.getLBPHist(testImage, P=8, R=1)

normHist = hist - xMean / (xStd + EPSILON)

prediction = classifier.predict([normHist])
print(prediction)

plt.show()