import cv2 as cv
import mahotas

import random

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

    return fd_hog(grad, 16)


# non usato
def contours(image_):
    imgray = cv.cvtColor(image_, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    _, contours_, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    new_image = cv.drawContours(image_, contours_, -1, (0, 255, 0), 3, )

    return new_image


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

    glob_features_b = np.hstack([f_moments, f_haralick, f_histogram, f_hog,
                                 features_vertical_hog, features_horizontal_hog])
    return glob_features_b

# cluster di keypoints di un'immagine trovati tramite il metodo SIFT
# non usati
def euclDistance(vector1, vector2):
    return sum(abs(vector2 - vector1))


def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k + 1, dim))
    s = set()
    for i in range(1, k + 1):
        while True:
            index = int(random.uniform(0, numSamples))
            if index not in s:
                s.add(index)
                break
        # index = int(random.uniform(0, 2))
        # print "random index:"
        # print index
        centroids[i, :] = dataSet[index, :]
    return centroids


def kmeans(img, k):
    # 　inizializzo un oggetto SIFT
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kp, dataSet = sift.detectAndCompute(gray, None)
    pic = cv.drawKeypoints(gray, kp, img)

    numSamples = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    for i in range(numSamples):
        clusterAssment[i, 0] = -1
    clusterChanged = True
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(1, k + 1):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist
            else:
                clusterAssment[i, 1] = minDist

        for j in range(1, k + 1):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids.flatten(), clusterAssment


def cluster_centroid(image_):
    k = 50
    centroids, clusterAssment = kmeans(image_, k)
    result = np.zeros(k, dtype=np.int16)
    for i in range(clusterAssment.shape[0]):
        categories = int(clusterAssment[i, 0] - 1)
        # print categories
        result[[categories]] += 1

    return result
