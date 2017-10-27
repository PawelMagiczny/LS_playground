'''
This module is responsible for loading data
'''
from skimage import io
import numpy as np

def load_image(file_path):
    '''
    Loads an image (usually) from data/ and returns an numpy ndarray
    INPUT:
    file_path - string pointing to an image.
    OUTPUT:
    img - numpy ndarray object representing an image
    FORMATS:
    *png, *bmp
    '''
    possible_formats = ['bmp', 'png']
    assert file_path[-3:] in possible_formats
    img = io.imread(file_path)
    assert isinstance(img, np.ndarray)
    assert np.prod(img.shape) > 100
    return img
