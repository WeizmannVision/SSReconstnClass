from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout, Cropping2D,Subtract,Conv3D,Activation,Reshape,AveragePooling2D,UpSampling2D,Concatenate, Conv2DTranspose
from keras.models import Model, Sequential
from keras.regularizers import l2,l1_l2,l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Add
from keras.initializers import RandomNormal
from keras.activations import relu
from keras import layers

from Models.layers import *
from Models.group_norm import GroupNormalization

class decoder_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.conv_ch = 64
        self.conv_l1_reg = 1e-5
        self.conv_l2_reg = 1e-4
        self.fc_reg_l1 = 20#1e-7#20
        self.fc_reg_gl = 400#5e-5#400
        self.fc_reg_gl_n = 0.5
        self.num_conv_layers = 3
        self.conv1_dim  =14
        self.out_ch =3
        self.resolution = 112
        self.num_fmris_per_segment = 3
        self.num_frames_per_segment = 15
        self.out_act = 'sigmoid'
        self.ker_size = 5


def decoder(param,W =None,name = 'decoder', bn = False):
    input_shape = (param.num_voxels,)
    model = Sequential(name = name)
    model.add(dense_f2c_gl( out=[param.conv1_dim,param.conv1_dim,param.conv_ch],l1=param.fc_reg_l1,l2=param.fc_reg_gl,n_reg=param.fc_reg_gl_n,W=W,input_shape = input_shape,name='fc_d') )
    model.add(Reshape((param.conv1_dim, param.conv1_dim, param.conv_ch)))

    for i in range(param.num_conv_layers):
        if(i>0):
            conv_ch = param.conv_ch
        else:
            conv_ch = param.conv_ch
        model.add(UpSampling2D((2, 2),interpolation='bilinear'))
        model.add(Conv2D(conv_ch, (param.ker_size, param.ker_size), padding='same', kernel_initializer="glorot_normal", activation='relu',
                         kernel_regularizer=l1(param.conv_l1_reg)))
        if bn == False:
            model.add(GroupNormalization(groups= int(param.conv_ch/2), axis=-1))
        else:
            model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(param.out_ch, (param.ker_size, param.ker_size), padding='same', kernel_initializer="glorot_normal", activation=param.out_act,
                     kernel_regularizer=l1(param.conv_l1_reg)))
    if(param.out_act == 'tanh'):
        model.add(Lambda(lambda img: (img+1)/2))

    return model
