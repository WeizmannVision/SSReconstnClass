#######################################################################################################################
#                                                   encoder_train.py                                                  #
# This script trains an encoder model (Image-to-fMRI). The encoder is trained in a fully supervised way.              #
# The output of this script is the weights of the trained encoder (saved in res_dir).                                 #
#######################################################################################################################


import tensorflow as tf
import os
import scipy.stats as stat
import sklearn.model_selection as sk_m
import sklearn.preprocessing

import copy
import sys
import math
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop
from keras.losses import mean_squared_error,cosine_proximity,mean_absolute_error
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.backend import log

from Utils.gen_functions import calc_snr
from Models.encoder_model import *
from Utils.batch_generator import *

from fMRIonImageNet.data.kamitani_data_handler import kamitani_data_handler as data_handler
import numpy as np

# Inputs
res_dir = str(sys.argv[1])
gpu     = str(sys.argv[2])
type = int(sys.argv[3])
subject = int(sys.argv[4])


os.environ["CUDA_VISIBLE_DEVICES"] = gpu#"1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from Utils.callbacks import *
from Utils.image_loss import *

from Models.layers import c2f_map, c2f_map_const_init
from Models.locally_connected import locally_connected_1d

###### configs - change save dirs ##############
dir ='data/Kamitani_Files/'
save_weights_file = res_dir+'encoder_weights0.hdf5' # None#dir+'models/save_encoder_sub5_seperable.hdf5'

# Learning rate control
def step_decay(epoch):
    lrate = 0.001
    if(epoch>20):
        lrate = 0.0001
    if (epoch > 30):
        lrate = 0.00001
    if (epoch > 35):
        lrate = 0.000001
    if (epoch > 50):
        lrate = 0.0000001

    return lrate

name = 'encoder_ml'   # used for save dirs

print(name)

# Get voxel data (with the possibility of taking just a specific region)

handler = data_handler(matlab_file = dir+'Subject'+str(subject)+'.mat')
Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_VC')

if type == 6:
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_V1')

if type == 7:
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_LVC')

if type == 8:
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_HVC')

labels_train, labels = handler.get_labels()

ROI_LVC = handler.get_meta_field('ROI_LVC').astype(bool)
ROI_HVC = handler.get_meta_field('ROI_HVC').astype(bool)


NUM_VOXELS = Y.shape[1]

# Get image data
dir = 'data/ImageNet_Files/'
file= np.load(dir+'images/images_112.npz')
X = file['train_images']
X_test = file['test_images']

X= X[labels_train]
X_test_sorted = X_test
X_test = X_test[labels]


# Define voxel loss
def combined_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) +  0.1*cosine_proximity(y_true, y_pred)


# Get models and set parameters
enc_param = encoder_param(NUM_VOXELS)
vision_model = encoder_ml_seperable(enc_param,vgg_loss,ch_mult = 1)
vision_model.compile(loss=combined_loss, optimizer= Adam(lr=1e-3,amsgrad=True), metrics=['mse','cosine_proximity','mae'])

reduce_lr = LearningRateScheduler(step_decay)


# corr_cb = corr_metric_callback(train_data=[X,Y],test_data=[X_test_sorted,Y_test_avg],num_voxels = NUM_VOXELS,ROI_LVC=ROI_LVC,ROI_HVC=ROI_HVC)

train_generator = batch_generator_enc(X, Y, batch_size=64,max_shift = 5)
test_generator = batch_generator_enc(X_test_sorted, Y_test_avg, batch_size=50,max_shift = 0)

vision_model.fit_generator(train_generator, steps_per_epoch=1200//64 , validation_steps=1 , epochs=50,validation_data=test_generator ,verbose=2,use_multiprocessing=False,callbacks=[reduce_lr])

if(save_weights_file is not None):
    vision_model.save_weights(save_weights_file)