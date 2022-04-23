#######################################################################################################################
#                                                   encdec_train.py                                                   #
# This script trains an decoder model (fMRI-to-Image).                                                                #
# The decoder is trained using pairs of {Image,fMRI}, as well as unpaired images and fMRIs.                           #
# For the self-supervision, a pretrained encoder is necessary (should be generated with encoder_train.py).            #
# The outputs of this script are: (i) Image reconstructions from test fMRI and (ii) decoder weights.                  #
#######################################################################################################################

import os

import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# Inputs
res_dir = str(sys.argv[1])
gpu     = str(sys.argv[2])
stage = int(sys.argv[3])
type = int(sys.argv[4])
repeat = int(sys.argv[5])
subject = int(sys.argv[6])


os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_unpaired = 16
batch_paired = 16

os.environ["CUDA_VISIBLE_DEVICES"] = gpu#"1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from keras.models import Model
from keras.callbacks import LearningRateScheduler

from Utils.save_images import save_model_results
from fMRIonImageNet.data.kamitani_data_handler import kamitani_data_handler as data_handler
from Utils.gen_functions import calc_snr

from Models.encoder_model import *
from Models.decoder_model import *
from Utils.image_loss import *
from Utils.batch_generator import *
from Utils.callbacks import *

# cfgs
num_ext_per_class = 50
scales[0:6] = [0, 1, 1, 1, 1, 1] # Perceptaul Similarity layers
ignore = None

# For Perceptual Similarity Layers Ablation
if type == 9:
    if repeat == 0:
        scales[0:6] = [0, 1, 0, 0, 0, 0]
    if repeat == 1:
        scales[0:6] = [0, 1, 1, 0, 0, 0]
    if repeat == 2:
        scales[0:6] = [0, 1, 1, 1, 0, 0]
    if repeat == 3:
        scales[0:6] = [0, 1, 1, 1, 1, 0]
    if repeat == 4:
        scales[0:6] = [0, 1, 1, 1, 1, 1]


dir ='data/Kamitani_Files/'

name = 'encdec_stage_'+str(stage)+'_type_'+str(type)  + '_repeat_' +str(repeat)  # used for save dirs
encoder_weights = res_dir+'encoder_weights'+str(stage-1)+'.hdf5'

# Define image loss
def feature_loss(y_true, y_pred ):
        return 10*vgg_total_loss(y_true,y_pred)  + mae(y_true,y_pred)

def combined_loss(y_true, y_pred):
    return feature_loss(y_true, y_pred)+  Tv_reg *total_variation_loss(y_pred)

#################################################### data load ##########################################################


# Get voxel data (with the possibility of taking just a specific region)
handler = data_handler(matlab_file = dir+'Subject'+str(subject)+'.mat')

Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_VC', imag_data=0)

if type == 6: #V1 only
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_V1')

if type == 7: #LVC only
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_LVC')

if type == 8: #HVC only
    Y, Y_test, Y_test_avg = handler.get_data(roi='ROI_HVC')


labels_train, labels = handler.get_labels(imag_data = 0)

Y_test_median = Y_test_avg

# Get image data

dir ='data/ImageNet_Files/'
file= np.load(dir+'images/ext_images_test_112.npz')
ext_img_test = file['img_112']

file= np.load(dir+'images/images_112.npz') #_56

X = file['train_images']
X_test = file['test_images']
X_test_sorted = X_test

X= X[labels_train]
X_test = X_test[labels]

NUM_VOXELS = Y.shape[1]
print(NUM_VOXELS)
RESOLUTION = 112

snr  = calc_snr(Y_test,Y_test_avg,labels)
snr_inv = 1/snr

snr = snr/snr.mean()
snr_inv = snr_inv/snr_inv.mean()


SNR  = tf.constant(snr,shape = [1,len(snr)],dtype = tf.float32)

# Define voxel loss
def mse_vox(y_true, y_pred):
    return K.mean(SNR*K.square(y_true-y_pred),axis=-1)

def mae_vox(y_true, y_pred):
    return K.mean(SNR*K.abs(y_true-y_pred),axis=-1)

def combined_voxel_loss(y_true, y_pred):
    return mae_vox(y_true, y_pred) +  0.1 *cosine_proximity(y_true, y_pred)

def maelog_vox(y_true, y_pred):
    return K.mean(SNR*K.log(K.abs(y_true-y_pred)+1),axis=-1)


# Learning rate control

initial_lrate = 0.001
epochs_drop =  30.0

def step_decay(epoch):

   drop = 0.2

   if math.floor((1+epoch)/epochs_drop) == 3:
       lrate = initial_lrate * math.pow(drop, 2)
   elif math.floor((1+epoch)/epochs_drop) == 4:
       lrate = initial_lrate * math.pow(drop, 3)
   elif math.floor((1 + epoch) / epochs_drop) == 5:
       lrate = initial_lrate * math.pow(drop, 4)
   else:
        lrate = initial_lrate * math.pow(drop,
            math.floor((1+epoch)/epochs_drop))
   return lrate

epochs = int(epochs_drop * 5)

print(name)


# Define encoder-decoder and decoder-encoder model (for self-supervision)
def encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model):

    input_voxel = Input((NUM_VOXELS,))
    input_img = Input((RESOLUTION, RESOLUTION, 3))
    input_mode = Input((1,))

    pred_voxel = encoder_model(input_img)
    rec_img_dec = decoder_model(input_voxel)

    rec_img_encdec = decoder_model(pred_voxel)
    pred_voxel_decenc = input_voxel # remove DE path

    if type == 0:
        if(repeat == 2 or repeat == 1):
            print('no ED')
            rec_img_encdec = input_img


    out_rec_img = Lambda(lambda t: K.switch(t[0],t[1],t[2]) ,name = 'out_rec_img') ([input_mode,rec_img_dec,rec_img_encdec])
    out_pred_voxel = Lambda(lambda t: K.switch(t[0], t[1], t[2]), name='out_pred_voxel')(
            [input_mode, pred_voxel, pred_voxel_decenc])

    return Model(inputs=[input_voxel,input_img,input_mode],outputs=[out_rec_img,out_pred_voxel]) #,out_pred_voxel


vgg_loss.calc_norm_factors(X)

# Get models and set parameters
dec_param = decoder_param(NUM_VOXELS)
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.25
encoder_model = encoder_ml_seperable(enc_param,vgg_loss,ch_mult = 1)
decoder_model = decoder(dec_param,W=snr_inv)
encoder_model.trainable = False
encoder_model.load_weights(encoder_weights)
#
#

model = encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model)


model.compile(loss= {'out_rec_img':combined_loss,'out_pred_voxel':combined_voxel_loss},loss_weights=[1.0,1.0],optimizer=Adam(lr=5e-4,amsgrad=True),metrics={'out_rec_img':['mse','mae']}) #,'out_pred_voxel':['mse','cosine_proximity','mae']  #,'out_pred_voxel':combined_voxel_loss

#####################################################model access functions ###############################################

def pred_dec(model,y):
    return model.predict( [y,np.zeros([y.shape[0],RESOLUTION,RESOLUTION,3]),np.ones([y.shape[0],1]) ],batch_size=50)[0]
def pred_enc(model,x):
    return model.predict( [ np.zeros([x.shape[0],NUM_VOXELS]),x, np.ones([x.shape[0], 1])], batch_size=50)[1]
def pred_encdec(model,x):
    return model.predict([np.zeros([x.shape[0],NUM_VOXELS]), x, np.zeros([x.shape[0], 1])], batch_size=50)[0]



##################################################### callbacks ###########################################################

reduce_lr = LearningRateScheduler(step_decay)

collage_cb = log_image_collage_callback(Y_test_avg, X_test_sorted, decoder_model, dir = res_dir+name+'/test_collge_ep/')

loader_train = batch_generator_encdec(X, Y, np.zeros(Y_test.shape), labels, batch_paired = batch_paired, batch_unpaired = batch_unpaired, num_ext_per_class=num_ext_per_class, ignore_test_fmri_labels=ignore)

model.fit_generator(loader_train, epochs=epochs, verbose=2,callbacks=[reduce_lr, collage_cb],workers=3,use_multiprocessing=True) #epochs

res_dir = res_dir+name+'/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

decoder_model.save_weights(res_dir + 'decoder_weights' + str(stage) + '.hdf5')

save_model_results(model,Y,X,Y_test,X_test_sorted,labels,Y_test_avg,Y_test_median,X_ext_full=ext_img_test,X_ext_part=ext_img_test,folder = res_dir,pred_func =pred_dec,pred_func_enc_dec =pred_encdec)#,runavg_callback = runavg_callback)



