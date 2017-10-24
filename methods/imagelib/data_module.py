from skimage import io
import numpy

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
    img = io.imread(file_path)
    assert type(img) == numpy.ndarray
    return img


