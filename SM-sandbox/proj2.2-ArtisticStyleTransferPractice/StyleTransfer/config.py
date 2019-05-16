import numpy as np


## Inputs 
file_content_image = 'img/schuster_5.jpg' 
file_style_image = 'img/starry_night_1.jpg'   
content_shape =  (200,200)
style_shape =  (200,200)

## Parameters 
input_noise = 0     # proportion noise to apply to content image
weight_style = 0.02      # coefficient in range(0,1)
update_weight_at_checkpoint = 0

## Layers
layer_content = 'conv4_2' 
layers_style = ['conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
layers_style_weights = [0.2,0.2,0.2,0.2,0.2]
max_group = 5
#layers_style_weights = [0.1,0.2,0.2,0.3,0.2]

## VGG19 model
#  Downloaded from https://www.kaggle.com/teksab/imagenetvggverydeep19mat#imagenet-vgg-verydeep-19.mat
path_VGG19 = 'model/imagenet-vgg-verydeep-19.mat'
# VGG19 mean for standardisation (RGB)
VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

## Reporting & writing checkpoint images
# NB. the total # of iterations run will be n_checkpoints * n_iterations_checkpoint
n_checkpoints = 10000            # number of checkpoints
n_iterations_checkpoint = 20   # learning iterations per checkpoint
path_output = 'output'  # directory to write checkpoint images into
minimisation_routine = 'L-BFGS-B'