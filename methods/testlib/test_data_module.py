'''
Testing module for load_image module
'''
from methods.imagelib.data_module import load_image

import numpy as np

import pytest

def test_load_image():
    '''
    Checks if
    - image is in possible formats (*bmp, *png)
    - input file is not empty
    - rises error when given non image
    - return type is numpy ndarray
    '''
    # input should have a valid file format
    file_path = 'data/2d/jhony_test/test1.jpg'
    with pytest.raises(AssertionError):
        load_image(file_path)
    # input should be an image
    file_path = 'data/2d/jhony_test/dupa.png'
    with pytest.raises(IOError):
        load_image(file_path)
    file_path = 'data/2d/jhony_test/dupa1.png'
    with pytest.raises(IOError):
        load_image(file_path)
    # input should exist
    file_path = 'data/2d/jhony_test/i_dont_exist.bmp'
    with pytest.raises(IOError):
        load_image(file_path)
    # output should be np.ndarray
    file_path = 'data/2d/jhony_test/test1.bmp'
    assert isinstance(load_image(file_path), np.ndarray)
    # input should not be empty file
    file_path = 'data/2d/jhony_test/empty.png'
    with pytest.raises(IOError):
        load_image(file_path)
    # output should have at least a few pixels
    file_path = 'data/2d/jhony_test/test4.png'
    img = load_image(file_path)
    assert np.prod(img.shape) > 100
