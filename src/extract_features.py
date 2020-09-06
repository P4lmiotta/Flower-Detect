import os
import cv2 as cv
import mahotas
from skimage import feature

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

'''
in seguito sono definite 3 funzioni che calcoleranno le features da dare al training set
'''


# il momento è una determinata media ponderata delle intensità dei pixel dell'immagine
def fd_hu_moments(image_):
    # image_ = cv.resize(image_, (500, 500))
    image_ = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image_)).flatten()
    return feature


# Haralick Texture viene utilizzato per quantificare un'immagine in base alla texture
def fd_haralick(image_):
    # image_ = cv.resize(image_, (500, 500))
    # conver the image to grayscale
    gray = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# Gli istogrammi sono dei conteggi raccolti di dati organizzati in una serie di contenitori predefiniti
def fd_histogram(image_, mask=None):
    # image_ = cv.resize(image_, (500, 500))
    bins = 8  # contenitori per un range di valori pari a 256
    # conver the image to HSV colors-space
    image_ = cv.cvtColor(image_, cv.COLOR_BGR2HSV)
    # COPUTE THE COLOR HISTPGRAM
    hist = cv.calcHist([image_], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv.normalize(hist, hist)
    # return the histog....
    return hist.flatten()


def fd_edge_detector(image_):
    image_ = cv.resize(image_, (150, 150))
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    # Load the image
    src = cv.GaussianBlur(image_, (3, 3), 0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad.flatten()


def fd_hog(image_):
    image_ = cv.resize(image_, (150, 150))
    image_ = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
    (hog, hog_image) = feature.hog(image_, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualise=True)

    return hog.flatten()


def extract(img):
    f_moments = fd_hu_moments(img)
    f_haralick = fd_haralick(img)
    f_histogram = fd_histogram(img)
    # f_hog = fd_hog(img)  # non usato
    # f_grad = fd_edge_detector(img)  # non usato

    glob_features_b = np.hstack([f_moments, f_haralick, f_histogram])
    return glob_features_b


# test features su un'immagine
if __name__ == '__main__':
    tulip = 'C:\\Users\\pc\\PycharmProjects\\pythonProject\\flowers_detection\\datasets\\flowers\\tulip\\112428665_d8f3632f36_n.jpg'
    rose = 'C:\\Users\\pc\\PycharmProjects\\pythonProject\\flowers_detection\\datasets\\flowers\\rose\\102501987_3cdb8e5394_n.jpg'

