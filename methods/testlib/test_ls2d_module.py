'''
Testing module for 2dls_module.py
'''
from methods.levelsetlib.ls2d_module  import calc_velocity_field
from methods.levelsetlib.ls2d_module  import init_phi
from methods.levelsetlib.ls2d_module  import perform_ls

from methods.imagelib.data_module import load_image


import numpy as np
import pytest

def test_calc_velocity_field():
    '''
    Checks if image is np.ndarray
    '''
    # input should be np.ndarray
    with open('data/2d/jhony_test/test1.bmp', 'r') as file_img:
        img_str = file_img.read()
    with pytest.raises(AssertionError):
        calc_velocity_field(img_str)
    # correct input should not rise errors
    img = np.ones((5, 55))
    calc_velocity_field(img)

def test_init_phi():
    '''
    Checks if image is np.ndarray
    and if phi has same shape
    '''
    # input should be np.ndarray
    with open('data/2d/jhony_test/test1.bmp', 'r') as file_img:
        img_str = file_img.read()
    with pytest.raises(AssertionError):
        init_phi(img_str)
    # correct input should not rise errors
    img = np.ones((5, 55))
    init_phi(img, dist_from_border=4, inside_val=1.0, outside_val=-1.0)
    # phi should have the same shape as image
    img = np.ones((10, 20))
    phi = init_phi(img, dist_from_border=4,
                   inside_val=1321.0, outside_val=-3211.0)
    assert img.shape == phi.shape

def test_perform_ls():
    '''
    Checks if return value (phi) has the same shape as image
    '''
    img = load_image('data/2d/jhony_test/test2.bmp')
    phi_final = perform_ls(img, learning_rate=1,
                           iterations=500, print_steps=False)
    assert img.shape[:2] == phi_final.shape
