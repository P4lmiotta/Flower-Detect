from skimage.transform import rescale
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage import util
from skimage.transform import rotate
from scipy import ndimage

'''
Questo file permette di modificare l'immagine in modo da avere più esempi nel dataset.
In un sviluppo futuro potrà essere utilizzato per fare data augmentation.
'''

def rescale_img(original_image):
    image_rescaled = rescale(original_image, 1.0 / 4.0)
    return image_rescaled

def add_noise(original_image):
    image_with_random_noise = random_noise(original_image)
    return image_with_random_noise

def gray_image(original_image):
    gray_scale_image = rgb2gray(original_image)
    return gray_scale_image

def color_inversion(original_image):
    color_inversion_image = util.invert(original_image)
    return color_inversion_image

def rotate_image(original_image):
    image_with_rotation = rotate(original_image, 45)
    return image_with_rotation

def horizontal_flip(original_image):
    horizontal_flip_ = original_image[:, ::-1]
    return horizontal_flip_

def vertical_flip(original_image):
    vertical_flip_ = original_image[::-1, :]
    return vertical_flip_

def blur(original_image):
    blurred_image = ndimage.uniform_filter(original_image, size=(11, 11, 1))
    return blurred_image
