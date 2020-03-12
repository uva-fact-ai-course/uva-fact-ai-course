"""
Code by Oscar Li with pytorch-compatible changes
github.com/OscarcarLi/PrototypeDL
"""
import torch
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    """
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].

    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields

    returns: an elastically distorted image of the same shape
    """
    images = images.squeeze(1)
    images = images.view(len(images), 28*28)
    assert len(images.shape) == 2

    # the two lines below ensure we do not alter the array images
    e_images = torch.zeros(images.shape)
    e_images[:] = images

    e_images = e_images.reshape(-1, height, width)

    if random_state is None:
        random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]

    for i in range(e_images.shape[0]):

        dx = gaussian_filter(
            (random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter(
            (random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        e_images[i] = torch.from_numpy(map_coordinates(e_images[i], indices, order=1))

    return e_images.reshape(-1, 1, 28, 28)
