'''
Testing module for preproc module
'''
from methods.imagelib.preproc_module import convert2grayscale
from methods.imagelib.preproc_module import normalize_image
from methods.imagelib.preproc_module import smooth_image

import numpy as np
import pytest

def test_convert2grayscale():
    '''
    Checks if input is an image (numpy.ndarray)
    '''
    # input should be np.ndarray
    with open('data/2d/jhony_test/test1.bmp', 'r') as file_img:
        img_str = file_img.read()
    with pytest.raises(AssertionError):
        convert2grayscale(img_str)
    # correct input should rise no errors
    img = np.ones((64, 64))
    convert2grayscale(img)

def test_normalize_image():
    '''
    Checks if input is an image (numpy.ndarray)
    '''
    # input should be np.ndarray
    with open('data/2d/jhony_test/test2.bmp', 'r') as file_img:
        img_str = file_img.read()
    with pytest.raises(AssertionError):
        normalize_image(img_str)
    # cannot rescale when min intensity = max intensity
    img = np.ones((64, 64))
    with pytest.raises(AssertionError):
        normalize_image(img, mean_flag=True, rescale_flag=True)
    # correct input should rise no errors
    img = np.ndarray((2, 2))
    img[0, :] = [2, 6]
    img[1, :] = [2.3, 666]
    normalize_image(img, mean_flag=True, rescale_flag=True)
    # no flags returns same image
    img = np.ndarray((2, 2))
    img[0, :] = [2, 6]
    img[1, :] = [2.3, 666]
    assert img.tolist()\
        == normalize_image(img, mean_flag=False, rescale_flag=False).tolist()

def test_smooth_image():
    '''
    Checks if input is an image (numpy.ndarray)
    '''
    # input should be np.ndarray
    with open('data/2d/jhony_test/test1.bmp', 'r') as file_img:
        img_str = file_img.read()
    with pytest.raises(AssertionError):
        smooth_image(img_str)
    # correct input should rise no errors
    img = np.ones((64, 64))
    smooth_image(img)
