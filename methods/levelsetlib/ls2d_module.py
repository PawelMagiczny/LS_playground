'''
Module responsible for performing level sets
'''
from methods.imagelib.preproc_module import convert2grayscale
from methods.imagelib.preproc_module import normalize_image
from methods.imagelib.preproc_module import smooth_image

import numpy as np
import matplotlib.pyplot as plt

def init_phi(img, dist_from_border=5, inside_val=1.0, outside_val=-1.0):
    '''
    Initializes the surface phi <dist_from_border> from the border of the image
    INPUT:
    img - 2d image (with or without colors) in the form of np.ndarray
    dist_from_border - tells you the distance from border of the image
    inside_val - value outside the curve
    outside_val - value inside the curve
    '''
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 3)
    assert dist_from_border <= np.min(img.shape[:2])
    phi = np.zeros(img.shape[:2])
    phi[:, :] = outside_val
    phi[dist_from_border:-dist_from_border, dist_from_border:-dist_from_border]\
        = inside_val
    return phi

def calc_velocity_field(img):
    '''
    Calculates the velocity field of the image (??)
    formula: 1 / 1 + grad(img)^2
    INPUT:
    img - image in a form of np.ndarray
    OUTPUT:
    velocity_field - (Jhony) force that drives curve evolution
        it is a vector field derived from the image where at every point
        it tells us the direction and magnitude of movement of our surface phi
    '''
    assert isinstance(img, np.ndarray)
    velocity_field\
        = 1.0 / (1.0 + np.linalg.norm(np.gradient(img), axis=0)**2)
    assert velocity_field.shape == img.shape
    return velocity_field

def perform_ls(img, learning_rate=1, iterations=500,
               print_steps=False):
    '''
    Performs level set according to update formula:
    phi = phi + learning_rate * velocity_field * d phi_norm
    INPUT:
    img - image on which we will perform level-sets
            can be normalized or smoothed
    learning_rate - ??
    iterations - nr of iterations/steps to perform
    print_steps - if true, it prints embedding function (?)
                  at every step of the iteration
    OUTPUT:
    phi - final value of the embedding function
    '''
    # preprocess image
    img = convert2grayscale(img)
    img = normalize_image(img)
    img = smooth_image(img)
    # initialize phi (level set function ie signed dist function)
    phi = init_phi(img, dist_from_border=5, inside_val=1.0, outside_val=-1.0)
    #print("!!!!!!!!!!!!!!!!!!! shape: {}".format(phi.shape))
    # calculate velocity field
    velocity_field = calc_velocity_field(img)
    if print_steps:
        plt.figure(1)
    # iterate to evolve the PDE
    for _ in range(iterations):
        # calculate gradient of phi
        dphi = np.gradient(phi)
        assert dphi[0].shape == phi.shape
        assert dphi[1].shape == phi.shape
        # calculate norm of a grad(phi)
        dphi_norm = np.linalg.norm(dphi, axis=0)
        assert dphi_norm.shape == phi.shape
        # level set PDE
        phi = phi - learning_rate * velocity_field * dphi_norm
        if print_steps:
            plt.clf()
            # plot the zero level curve of phi
            plt.contour(phi, 0)
            plt.draw()
            plt.pause(0.05)
    # print("after!!!!!!!!!!!!!!!!!!! shape: {}".format(phi.shape))

    return phi
