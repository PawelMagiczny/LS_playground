# Simple image segmentation using the level set method
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation
from skimage import color, io
from skimage.exposure import rescale_intensity

# read an image
#img = io.imread('Z:/dev/level-set/test.bmp')
img = io.imread('data/2d/jhony_test/test4.png')

# convert the image to grayscale
img_gray = color.rgb2gray(img)
# normalize by the mean
img_norm = img_gray - np.mean(img_gray)
img_norm = rescale_intensity(img_norm, out_range=(0, 255))
# smooth the image to reduce noise
sigma = 2
img_smooth = scipy.ndimage.filters.gaussian_filter(img_norm, sigma)


# simple edge detector by taking the gradients of the image
def velocity_field(x):
    return 1. / (1. + np.linalg.norm(np.gradient(x), axis=0)**2)

# force that drive curve evolution (i.e. the velocity field)
# it is a vector field derived from the image where at every point it
# tells us the direction and magnitude of movement of our surface phi
F = velocity_field(img_smooth)

def init_phi(x):
    # initialize the surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi

# iterate to evolve the PDE
plt.figure(1)
phi = init_phi(img_smooth) # level set function (i.e. signed distance function)
dt = 1 # learning rate
n_iter = 500
for i in range(n_iter):
    plt.clf()
    dphi = np.gradient(phi)
    dphi_norm = np.linalg.norm(dphi, axis=0)
    # level set PDE
    phi = phi + dt * F * dphi_norm
    # plot the zero level curve of phi
    plt.contour(phi, 0)
    plt.draw()
    plt.pause(0.05)

# plot everything
fig, ax = plt.subplots(2, 3)
ax[0,0].imshow(img, cmap='gray')
ax[0,1].imshow(img_gray, cmap='gray')
ax[0,2].imshow(img_norm, cmap='gray')
ax[1,0].imshow(img_smooth, cmap='gray')
ax[1,1].imshow(F, extent=[0, 1, 0, 1])
ax[1,2].imshow(img, cmap='gray')
ax[1,2].contour(phi, 0, colors='r')
plt.show()

fig2 = plt.figure()
plt.imshow(phi)
plt.show()
