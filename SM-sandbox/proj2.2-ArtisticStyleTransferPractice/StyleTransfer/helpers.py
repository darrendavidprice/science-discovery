import numpy as np
import tensorflow as tf
import scipy.misc, scipy.io
import os
from PIL import Image
from StyleTransfer.config import *


# Globals
VGG19_layers = None


### Helper functions
def imread(path,shape=None):
    img = Image.open(path)
    if shape is not None : img.thumbnail(shape)
    return np.asarray(img)
#    return scipy.misc.imread(path).astype(np.float)   # returns RGB format

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
    
def imgpreprocess(image):
    image = image[np.newaxis,:,:,:]
    return image - VGG19_mean

def imgunprocess(image):
    temp = image + VGG19_mean
    return temp[0]

def make_output_dir ( path ) :
    path = path + "-1"
    while os.path.exists(path):
        path_split = path.split("-")
        path = path[:-1*len(path_split[-1])] + str(int(path_split.pop())+1)
    os.mkdir(path)
    return path

# function to convert 2D greyscale to 3D RGB
def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def print_message ( *args ) :
    if len(args) == 0 : return
    ret = args[0]
    for arg in args[1:] :
        ret = ret + "\t\t" + arg
    print(ret)
    return ret

# help functions

def setup_VGG19 () :
    global VGG19, VGG19_layers
    VGG19 = scipy.io.loadmat(path_VGG19)
    VGG19_layers = VGG19['layers'][0]

def conv2d_relu(prev_layer, n_layer, layer_name):
    # get weights for this layer:
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    # create a conv2d layer
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
    # add a ReLU function and return
    return tf.nn.relu(conv2d)

def avgpool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


### STYLE LOSS: FUNCTION TO CALCULATE AND INSTANTIATION

def style_layer_loss(a, x):
    _, h, w, d = [i.value for i in a.get_shape()]
    M = h * w 
    N = d 
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, M, N):
    F = tf.reshape(x, (M, N))                   
    G = tf.matmul(tf.transpose(F), F)
    return G


# Recode to be simpler: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
def content_layer_loss(p, x):
    _, h, w, d = [i.value for i in p.get_shape()]    # d: number of filters; h,w : height, width
    M = h * w 
    N = d 
    K = 1. / (2. * N**0.5 * M**0.5)
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss