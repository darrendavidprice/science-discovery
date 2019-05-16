import numpy as np
import scipy.misc, scipy.io
import tensorflow as tf
from sys import stderr
from functools import reduce
import time
import math
from shutil import copyfile

from StyleTransfer.config import *
from StyleTransfer.helpers import *
 

####  ====================================== 
####  Preprocessing
####  ====================================== 

# create output directory
path_output = make_output_dir(path_output)
print_message("make_image.py","opened output directory:",path_output)
copyfile("StyleTransfer/config.py",path_output+"/0_config.py")

# read in images
print_message("make_image.py","reading content image:",file_content_image)
img_content = imread(file_content_image,content_shape) 
print_message("make_image.py","reading style image:",file_style_image)
img_style = imread(file_style_image,style_shape) 

# convert if greyscale
if len(img_content.shape)==2:
    print_message("make_image.py","converting content image to colour")
    img_content = to_rgb(img_content)

if len(img_style.shape)==2:
    print_message("make_image.py","converting style image to colour")
    img_style = to_rgb(img_style)

# resize style image to match content
print_message("make_image.py","resizing style image to match content")
img_style = scipy.misc.imresize(img_style, img_content.shape)

# apply noise to create initial "canvas" 
print_message("make_image.py","creating initial noise tensor")
noise = np.random.uniform(
        img_content.mean()-img_content.std(), img_content.mean()+img_content.std(),
        (img_content.shape)).astype('float32')
img_initial = noise * input_noise + img_content * (1 - input_noise)

# preprocess each
print_message("make_image.py","preprocessing images (?!)")
img_content = imgpreprocess(img_content)
img_style = imgpreprocess(img_style)
img_initial = imgpreprocess(img_initial)
imsave(path_output+'/0_content.jpg', imgunprocess(img_content))
imsave(path_output+'/0_style.jpg', imgunprocess(img_style))

####  ====================================== 
####  BUILD VGG19 MODEL
####  ====================================== 

## with thanks to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
print_message("make_image.py","loading model")
setup_VGG19()

# Setup network
print_message("make_image.py","setting up network")
with tf.Session() as sess:
    a, h, w, d     = img_content.shape
    net = {}
    net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
    net['conv1_1']  = conv2d_relu(net['input'], 0, 'conv1_1')
    net['conv1_2']  = conv2d_relu(net['conv1_1'], 2, 'conv1_2')
    if max_group > 1 :
        net['avgpool1'] = avgpool(net['conv1_2'])
        net['conv2_1']  = conv2d_relu(net['avgpool1'], 5, 'conv2_1')
        net['conv2_2']  = conv2d_relu(net['conv2_1'], 7, 'conv2_2')
    if max_group > 2 : 
        net['avgpool2'] = avgpool(net['conv2_2'])
        net['conv3_1']  = conv2d_relu(net['avgpool2'], 10, 'conv3_1')
        net['conv3_2']  = conv2d_relu(net['conv3_1'], 12, 'conv3_2')
        net['conv3_3']  = conv2d_relu(net['conv3_2'], 14, 'conv3_3')
        net['conv3_4']  = conv2d_relu(net['conv3_3'], 16, 'conv3_4')
    if max_group > 3 : 
        net['avgpool3'] = avgpool(net['conv3_4'])
        net['conv4_1']  = conv2d_relu(net['avgpool3'], 19, 'conv4_1')
        net['conv4_2']  = conv2d_relu(net['conv4_1'], 21, 'conv4_2')     
        net['conv4_3']  = conv2d_relu(net['conv4_2'], 23, 'conv4_3')
        net['conv4_4']  = conv2d_relu(net['conv4_3'], 25, 'conv4_4')
    if max_group > 4 : 
        net['avgpool4'] = avgpool(net['conv4_4'])
        net['conv5_1']  = conv2d_relu(net['avgpool4'], 28, 'conv5_1')
        net['conv5_2']  = conv2d_relu(net['conv5_1'], 30, 'conv5_2')
        net['conv5_3']  = conv2d_relu(net['conv5_2'], 32, 'conv5_3')
        net['conv5_4']  = conv2d_relu(net['conv5_3'], 34, 'conv5_4')
        net['avgpool5'] = avgpool(net['conv5_4'])


# with thanks to https://github.com/cysmith/neural-style-tf

print_message("make_image.py","defining content loss")
with tf.Session() as sess:
    sess.run(net['input'].assign(img_content))
    p = sess.run(net[layer_content])  # Get activation output for content layer
    x = net[layer_content]
    p = tf.convert_to_tensor(p)
    content_loss = content_layer_loss(p, x) 

print_message("make_image.py","defining style loss")
with tf.Session() as sess:
    sess.run(net['input'].assign(img_style))
    style_loss = 0.
    # style loss is calculated for each style layer and summed
    for layer, weight in zip(layers_style, layers_style_weights):
        a = sess.run(net[layer])
        x = net[layer]
        a = tf.convert_to_tensor(a)
        style_loss += style_layer_loss(a, x)

####  ====================================== 
####  Define loss function and minimise
####  ======================================

print_message("make_image.py","minimising total loss")
with tf.Session() as sess:
    # loss function
    L_total  = ((1-weight_style) * content_loss) + (weight_style * style_loss)
    
    # instantiate optimiser
    optimizer = tf.contrib.opt.ScipyOptimizerInterface( L_total, method=minimisation_routine, options={'maxiter': n_iterations_checkpoint})
    
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    sess.run(net['input'].assign(img_initial))

    ## write initial image
    print_message("make_image.py","printing initial image")
    imsave(path_output+'/0_initial.jpg', imgunprocess(sess.run(net['input'])))

    ## minimize!
    for i in range(1,n_checkpoints+1):
        # run optimisation
        optimizer.minimize(sess)
        
        ## print costs
        stderr.write('Iteration %d/%d\n' % (i*n_iterations_checkpoint, n_checkpoints*n_iterations_checkpoint))
        stderr.write('  content loss: %g\n' % sess.run((1.-weight_style) * content_loss))
        stderr.write('    style loss: %g\n' % sess.run(weight_style * style_loss))
        stderr.write('    total loss: %g\n' % sess.run(L_total))

        ## write image
        img_output = sess.run(net['input'])
        img_output = imgunprocess(img_output)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        output_file = path_output+'/'+timestr+'_'+'%s.jpg' % (i*n_iterations_checkpoint)
        print_message("make_image.py","printing updated image")
        imsave(output_file, img_output)

        ## update weight_style
        if update_weight_at_checkpoint is not None :
            if update_weight_at_checkpoint > -1 and update_weight_at_checkpoint < 1 :
                weight_style = weight_style + update_weight_at_checkpoint*(1-weight_style)

