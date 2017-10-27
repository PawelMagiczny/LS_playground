'''
This module is responsible for preprocessing images:
grayscale, normalizing, smoothing etc
'''
from skimage import color
from skimage.exposure import rescale_intensity
import numpy as np
from scipy.ndimage.filters import gaussian_filter
def convert2grayscale(img):
    '''
    Takes an image (numpy.ndarray) and converts it to grayscale
    INPUT:
    img - numpy.ndarray
    OUTPUT
    img_gray - image in grayscale
    '''
    assert isinstance(img, np.ndarray)
    img_gray = color.rgb2gray(img)
    assert img_gray.shape == (img.shape[0], img.shape[1])
    return img_gray

def normalize_image(img, mean_flag=True, rescale_flag=True):
    '''
    Takes an image and
    - normalizes by the mean
    - rescales intensity to the range (0,255)
    INPUT:
    img - image of typ np.ndarray
    OUTPUT:
    img_norm - normalized image
    '''
    assert isinstance(img, np.ndarray)
    # normalize by the mean
    if mean_flag:
        img_norm = img - np.mean(img)
    else:
        img_norm = img
    # rescale intensity to (0,255)
    if rescale_flag:
        # avoid division by 0
        assert np.min(img_norm) != np.max(img_norm)
        img_norm = rescale_intensity(img_norm, out_range=(0, 255))
        assert np.max(img_norm) <= 255
        assert np.min(img_norm) >= 0
    assert img_norm.shape == img.shape
    assert isinstance(img_norm, np.ndarray)
    return img_norm

def smooth_image(img, sigma=2):
    '''
    Smooths image using a gaussian filter
    INPUT:
    img - image to be smoothed type
    sigma - standard deviation for Gaussian kernel
    OUTPUT
    img_smooth - smoothed image
    '''
    assert isinstance(img, np.ndarray)
    img_smooth = gaussian_filter(img, sigma)
    assert img.shape == img_smooth.shape
    assert isinstance(img_smooth, np.ndarray)
    return img_smooth
