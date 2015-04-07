''' data_augmentation.py

'''
import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise


''' RGB PCA and variations from Alex's paper '''
def RGB_PCA(images):
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels)/256.
    C = np.cov(pixels)/(256.*256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    a = np.random.randn(3)
    v = np.array([a[0]*eig_val[0], a[1]*eig_val[1], a[2]*eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


def flip(x):
    return np.fliplr(x)


def rot(x, max_angle=30):
    a = max_angle*np.random.rand(1)[0]
    return rotate(x, a)


def noise(x):
    r = np.random.rand(1)[0]
    # TODO randomize parameters of the noises; check how to init seed
    if r < 0.33:
        return random_noise(x, 's&p', seed=np.random.randint(1000000))
    if r < 0.66:
        return random_noise(x, 'gaussian', seed=np.random.randint(1000000))
    return random_noise(x, 'speckle', seed=np.random.randint(1000000))
