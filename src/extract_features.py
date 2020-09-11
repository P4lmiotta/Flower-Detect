import cv2 as cv
import mahotas

from pylab import *

import src.data_augmentation

import numpy as np

'''
- In seguito sono definite una serie di funzioni per l'estrazione delle features delle immagini.
- Nel metodo extract verranno chiamate le funzioni dedicate all'estrazione delle features migliori
'''

# HuMoments è un metodo della libreria open-cv che permette di estrarre la forma degli oggetti presenti nell'immagine
def fd_hu_moments(image_):
    image_ = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image_)).flatten()
    return feature


# Haralick Texture viene utilizzato per quantificare un'immagine in base alla texture
def fd_haralick(image_):
    # converto l'immagine in una grayscale
    gray = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)
    # estraggo l'haralick features
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# Gli istogrammi sono dei conteggi raccolti di dati organizzati in una serie di contenitori, dedicata all'estarzione dei colori presenti nell'immagine
def fd_histogram(image_, mask=None):
    bins = 16  # contenitori per un range di valori pari a 2^16

    # converto l'immagine in un HSV colors-space
    image_ = cv.cvtColor(image_, cv.COLOR_BGR2HSV)

    # estraggo il color histogram
    hist = cv.calcHist([image_], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    # normalizzo l'istogramma
    cv.normalize(hist, hist)

    return hist.flatten()  # utilizzo il metodo flatten in modo da avere un array di una dimensione


def fd_hog(image_, bins):
    dx = cv.Sobel(image_, cv.CV_32F, 1, 0)
    dy = cv.Sobel(image_, cv.CV_32F, 0, 1)

    # Calcolo la magnitude e l'angolo
    magnitude, angle = cv.cartToPolar(dx, dy)

    # Quantifico i binvalues in (0..n_bins)
    binvalues = np.int32(bins * angle / (2 * np.pi))

    # Divido l'immagine in 4 parti
    magn_cells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]
    bin_cells = binvalues[:10, :10], binvalues[10:, :10], binvalues[:10, 10:], binvalues[10:, 10:]

    # Con "bincount" possiamo contare il numero di occorrenze di un
    # flat array per creare l'istogramma.
    histogram = [np.bincount(bin_cell.ravel(), magn.ravel(), bins)
                 for bin_cell, magn in zip(bin_cells, magn_cells)]

    return np.hstack(histogram)


# non usato
def grubcat(image_):
    mask = np.zeros(image_.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)
    cv.grabCut(image_, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image_ * mask2[:, :, np.newaxis]

    return img

# non usato
def fd_edge_detector(image_):
    image_ = cv.resize(image_, (500, 500))
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    src_g = cv.GaussianBlur(image_, (3, 3), 0)

    gray = cv.cvtColor(src_g, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    histogram = hist(grad.flatten(), 128)
    hist_ = np.hstack(histogram)

    return hist_.flatten()

# funzione per l'estrazione delle features, chiamato nel file 'classifier.py'
def extract(img):
    f_moments = fd_hu_moments(img)
    f_haralick = fd_haralick(img)
    f_histogram = fd_histogram(img)

    f_hog = fd_hog(img, 16)

    img_vert_flip = src.data_augmentation.vertical_flip(img)
    features_vertical_hog = fd_hog(img_vert_flip, 16)

    img_hori_flip = src.data_augmentation.horizontal_flip(img)
    features_horizontal_hog = fd_hog(img_hori_flip, 16)

    glob_features_b = np.hstack([f_moments, f_haralick, f_histogram,
                                 features_vertical_hog, features_horizontal_hog, f_hog])
    return glob_features_b
