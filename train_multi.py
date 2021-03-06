import os
import sys
import time
import argparse
from  multiprocessing import Pool, Queue

import cPickle as pickle
import numpy as np
import tables as tb
from skimage.transform import resize
from skimage.color import rgb2grey

import theano.tensor as T
import theano

from nn import *
from networks import *
from dataset_augmentation import *


# ----------------------------------------------------------------------------
# ARGUMENT PARSER


Parser = argparse.ArgumentParser(description='GPU Experiment')
Parser.add_argument('--db_path', default='/home/cesar/DB/dogs_vs_cats')
Parser.add_argument('--xp_path', default='TEST')
Parser.add_argument('--seed', type=int, default=77)
Parser.add_argument('--width', type=int, default=124)
Parser.add_argument('--height', type=int, default=124)
Parser.add_argument('--batch_size', type=int, default=100)
Parser.add_argument('--nepochs', type=int, default=1000)

Parser.add_argument('--data_augmentation', action='store_true', default=False)
Parser.add_argument('--rotate', action='store_true', default=False)
Parser.add_argument('--crop', action='store_true', default=False)
Parser.add_argument('--noise', action='store_true', default=False)
Parser.add_argument('--flip', action='store_true', default=False)
Parser.add_argument('--equalize', action='store_true', default=False)
Parser.add_argument('--grey', action='store_true', default=False)

Parser.add_argument('--lr', type=float, default=0.01) # 0.001
Parser.add_argument('--lr_decay', type=float, default=0.0) #0.00001
Parser.add_argument('--momentum_factor', type=float, default=0.0) #0.9
Parser.add_argument('--NAG', action='store_true', default=False)
Parser.add_argument('--L1_factor', type=float, default=0.0) #0.0001
Parser.add_argument('--L2_factor', type=float, default=0.0) #0.0001
Parser.add_argument('--stop_after', type=int, default=50)

# TODO: Finish dropout
Parser.add_argument('--dropout', action='store_true', default=False)
Parser.add_argument('--drop_prob', type=float, default=0.5)
Parser.add_argument('--drop_prob0', type=float, default=0.8)

Parser.add_argument('--arch', type=int, default=None)
Parser.add_argument('--kw0', type=int, default=9)
Parser.add_argument('--pool0', type=int, default=2)
Parser.add_argument('--nhu0', type=int, default=32)
Parser.add_argument('--kw1', type=int, default=9)
Parser.add_argument('--pool1', type=int, default=2)
Parser.add_argument('--nhu1', type=int, default=64)
Parser.add_argument('--kw2', type=int, default=6)
Parser.add_argument('--pool2', type=int, default=2)
Parser.add_argument('--nhu2', type=int, default=20) # 20 
Parser.add_argument('--kw3', type=int, default=6)
Parser.add_argument('--pool3', type=int, default=2)
Parser.add_argument('--nhu3', type=int, default=20) # 20 
Parser.add_argument('--kw4', type=int, default=6)
Parser.add_argument('--pool4', type=int, default=2)
Parser.add_argument('--nhu4', type=int, default=20) # 20 
Parser.add_argument('--nhu5', type=int, default=20) # 20 
Parser.add_argument('--nhu6', type=int, default=1000)

params = Parser.parse_args()


# -----------------------------------------------------------------------------
# PARAMETERS


# Theano params
theano.config.floatX = 'float32'
floatX = theano.config.floatX

# Init random number generator
np.random.seed(params.seed)

# DB parameters
params.TR_IDX = [0, 20000]
params.VA_IDX = [20000, 22500]
params.TE_IDX = [22500, 25000]
params.db = os.path.join(params.db_path, 'train.h5')

# Training parameters
if params.grey:
    params.batch_shape = (params.batch_size, 1, params.height, params.width)
else:
    params.batch_shape = (params.batch_size, 3, params.height, params.width)

# TODO experiment dir
if not os.path.exists(params.xp_path):
    os.makedirs(params.xp_path)
params.result_file = os.path.join(params.xp_path, 'results.txt')
params.final_result_file = os.path.join(params.xp_path, 'final_results.txt')
params.net_file = os.path.join(params.xp_path, 'net.bin')
params.best_net_file = os.path.join(params.xp_path, 'best_net.bin')

# Early stopping params
params.best_result = 1.
params.stop_counter = 0


# ----------------------------------------------------------------------------
# FUNCTIONS


def one_hot_encode(x, nclasses):
    y = np.zeros(nclasses)
    y[x] = 1
    return y


def load_dataset(params, s):
    if s == 'train':
        idx = params.TR_IDX
    elif s == 'valid':
        idx = params.VA_IDX
    elif s == 'test':
        idx = params.TE_IDX
    else:
        sys.exit('Unknown s parameter in load_dataset()!')
    with tb.open_file(params.db, 'r') as f:
        # Load labels into one hot encode vectors
        labels = []
        for x in f.root.Data.y.iterrows(idx[0], idx[1]):
            labels.append(one_hot_encode(x, 2))
        # Load the shapes of the images
        shapes = [x for x in f.root.Data.s.iterrows(idx[0], idx[1])]
        # Load the images
        out_file = '%s_data_%d.bin' %(s, 1.3*params.height)
        out_file = os.path.join(params.db_path, out_file)
        if not os.path.exists(out_file):
            images = np.zeros((idx[1]-idx[0],1.3*params.height, 1.3*params.height, 3),
                              dtype=np.uint8)
            for i, x in enumerate(f.root.Data.X.iterrows(idx[0], idx[1])):
                x = x.reshape(shapes[i])
                x = resize(x, (int(1.3*params.height), int(1.3*params.height), 3))
                x = (x*256).astype(np.uint8)
                images[i] = x
            with open(out_file, 'wb') as ff:
                images.tofile(ff)
        else:
            with open(out_file, 'rb') as ff:
                images = np.fromfile(ff, dtype=np.uint8)
                shape = (idx[1]-idx[0], 1.3*params.height, 1.3*params.height, 3)
                images = images.reshape(shape)
    # Prepare RGB PCA (for contrast data augmentation)
    if s == 'train':
        l, v, m = RGB_PCA(images)
        params.RGB_eig_val = l
        params.RGB_eig_vec = v
        params.RGB_mean = m
    return images, labels


def random_crop(image):
    x = np.random.randint(221, 256)
    image = resize(image, (x, x, 3))
    if x != 221:
        x, y = np.random.randint(0, 256-x)
        image = image[x:221+x, y:221+y, :]
    image = image.astype(floatX)
    image = np.rollaxis(image, 2, 0)
    return image


def data_augmentation(image, params):
    # 1. Resize and crop
    if params.crop:
        # Randomly zoom
        x = np.random.randint(params.height, params.height*1.2) - params.height
        image = resize(image, (params.height + x, params.width + x, 3))
        # Ranodmly crop
        if x != 0:
			x, y = np.random.randint(0, x, 2)
			image = image[x:params.height+x, y:params.width+y, :]
    else:
        image = resize(image, (params.height, params.width, 3))
    # Flip
    if params.flip and np.random.rand(1)[0] > 0.5:
        image = flip(image)
    # Normalize
    image = image.astype(floatX)
    # Equalize
    if params.equalize:
        image = RGB_variations(image, params.RGB_eig_val, params.RGB_eig_vec)
    # To greyscale
    if params.grey:
        image = rgb2grey(image)
    # Add noise
    if params.noise:
        image = noise(image)
    # Rotate
    if params.rotate:
        image = rot(image)
    else:
		image = rot(image, 0)
    # Remove mean
    image = image - params.RGB_mean # TODO 
    # Reshape for theano's convolutoins
    if params.grey:
        image = image.reshape(params.height, params.width, 1)
    image = np.rollaxis(image, 2, 0)
    return image


def pool_init(queue):
    prepare_batch.queue = queue


def prepare_batch(images, labels, params, action):
    # Allocate new batch
    x = np.zeros(params.batch_shape, dtype=floatX)
    t = np.zeros((params.batch_size, 2), dtype=floatX)
    # If training, fill batch with images that have been modified 
    if action == 'train' and params.data_augmentation:
        for b, img in enumerate(images):
            x[b] = data_augmentation(img, params)
            t[b] = labels[b]
    # If valid or test, fill batch with resized and normalized images
    else:
        for b, img in enumerate(images):
            image = resize(img, (params.height, params.width, 3))
            image = image.astype(floatX)
            if params.grey:
                image = rgb2grey(image)
                image = image.reshape(params.height, params.width, 1)
            image = image - params.RGB_mean # TODO
            x[b] = np.rollaxis(image, 2, 0)
            t[b] = labels[b]
    prepare_batch.queue.put((x, t))


def one_epoch(images, labels, net, pool, queue, params, action):
    # Multithreaded batches preparation
    idx = np.random.permutation(len(images))
    nbatches = len(idx)/params.batch_size
    workers = []
    for i in xrange(nbatches):
        indices = idx[i*params.batch_size:(1+i)*params.batch_size]
        imgs = [images[j].copy() for j in indices]
        labs = [labels[j] for j in indices]
        worker = pool.apply_async(prepare_batch, 
                                  args=(imgs, labs, params, action))
        workers.append(worker)
    # Network training
    err = 0.
    miss = 0.
    counter = 0    
    while True:
        x, t = queue.get()
        if params.dropout:
            drop(net, action, p=params.drop_prob, p0=params.drop_prob0)
        if action == 'train':
            L, M = net.train(x, t, params.lr)
        elif action == 'valid':
            L, M = net.predict(x, t)
        err += L
        miss += M
        counter += 1
        if counter == nbatches:
            break
    err /= nbatches
    miss /= nbatches
    counter = 0
    for worker in workers:
        if worker.ready():
            counter += 1
    assert counter == nbatches
    assert queue.empty()
    return err, miss


def save_model(network, epoch, params, t_err, t_miss, v_err, v_miss):
    with open(params.net_file, 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
    if v_miss < params.best_result:
        params.best_result = v_miss
        params.best_epoch = epoch
        params.stop_counter = 0
        params.best_train = t_err
        params.best_train_miss = t_miss
        params.best_valid = v_err
        params.best_valid_miss = v_miss
        with open(params.best_net_file, 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        params.stop_counter += 1


# ----------------------------------------------------------------------------
# MAIN LOOP


# Preparing network
print('Preparing Network')
network = create_network(params)
print('Done!')

# Prepare Pool and Queue
queue = Queue(20)
pool = Pool(3, pool_init, [queue]) 

# Loading dataset
print('Loading Data')
train_images, train_labels = load_dataset(params, 'train')
valid_images, valid_labels = load_dataset(params, 'valid')
print('Done!')

# Training loop
print('Training Loop')
for epoch in range(params.nepochs):
    # Performing train and valid epoch
    t = time.time()
    t_err, t_miss = one_epoch(train_images, train_labels, network, 
                              pool, queue, params, 'train')
    v_err, v_miss = one_epoch(valid_images, valid_labels, network, 
                              pool, queue, params, 'valid')

    # DEBUG TODO
    #print network.layers[7].get_bias().get_value()

    # Saving model
    save_model(network, epoch, params, t_err, t_miss, v_err, v_miss)
    
    # Printing results to console
    print('elapsed : ' + str(time.time() - t))
    t_acc = 100*(1 - t_miss)
    v_acc = 100*(1 - v_miss)
    s = '%d\t%.5f\t%.1f\t%.5f\t%.1f' %(epoch, t_err, t_acc, v_err, v_acc)
    print(s)
    
    # Saving results
    with open(params.result_file, 'a') as f:
        f.write(s + '\n')
    
    # Checking early stopping
    if params.stop_counter >= params.stop_after:
        print('    Early Stopping after ' + str(epoch) + ' epochs')
        break
    # Learning rate decay
    params.lr = lr_decay(params.lr, params.lr_decay, epoch)

# Testing with best network
del train_images
del train_labels
del valid_images
del valid_labels
del network

print('Loading Test Data')
test_images, test_labels = load_dataset(params, 'test')

print('Loading Best Network')
with open(params.best_net_file, 'rb') as f:
     params = pickle.load(f)
     network = pickle.load(f)
# TODO: replace this ugly hack
network.train, network.predict = network._compile_net(network.layers,
                                                      network.criterion,
                                                      params)

print('Testing Loop')
test_err, test_miss = one_epoch(test_images, test_labels, network,
                                pool, queue, params, 'valid')
# Printing final results
print('')
print('*' * 79)
print('Final results :')
se = 'Number of epochs for best valid performances : %d' %params.best_epoch
st = 'Train error (miss) : %.5f (%.5f)' %(params.best_train, 
                                          params.best_train_miss)
sv = 'Valid error (miss) : %.5f (%.5f)' %(params.best_valid, 
                                          params.best_valid_miss)
ste = 'Test  error (miss) : %.5f (%.5f)' %(test_err, 
                                           test_miss)
print(se)
print(st)
print(sv)
print(ste)
print('*' * 79)
with open(params.final_result_file, 'w') as f:
    f.write(se + '\n')
    f.write(st + '\n')
    f.write(sv + '\n')
    f.write(ste + '\n')
