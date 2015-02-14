''' data_augmentation.py

'''
import numpy as np
from skimage.transform import rotate, resize
from skimage.util import random_noise
from skimage.exposure import adjust_gamma


''' RGB PCA and variations from Alex's paper '''
def RGB_PCA(pixels):
    if pixels.shape[0] != 3:
        pixels = pixels.T
    pixels = pixels/255.
    m = np.mean(pixels)
    #pixels = pixels - m
    C = np.cov(pixels)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    a = np.random.randn(3)
    v = np.array([a[0]*eig_val[0], a[1]*eig_val[1], a[2]*eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


#def equalize(x, min_g=0.1, max_g=3):
    #gamma = max_g*(np.random.rand(1)[0]) + min_g
    #return adjust_gamma(x, gamma)


def flip(x):
    return np.fliplr(x)


def rot(x, max_angle=10):
    a = max_angle*np.random.rand(1)[0]
    return rotate(x, a)


def noise(x):
    r = np.random.rand(1)[0]
    # TODO randomize parameters of the noises; check how to init seed
    if r < 0.33:
        return random_noise(x, 's&p', seed=np.random.randint(10000))
    if r < 0.66:
        return random_noise(x, 'gaussian', seed=np.random.randint(10000))
    return random_noise(x, 'speckle', seed=np.random.randint(10000))


#def scale(x, params):
   # zoom_w = int((params.db_width - params.width)*np.random.rand(1)[0])
   # zoom_h = int(zoom_w*params.db_height/float(params.db_width))
   # p_w = int(zoom_w*np.random.rand(1)[0])
   # p_h = int(zoom_h*np.random.rand(1)[0])
   # x = x[p_h:params.height+p_h, p_w:params.width+p_w, :]
   # return x
