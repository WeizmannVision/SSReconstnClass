from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Dropout, Cropping2D,Subtract,Conv3D,Activation,Reshape,AveragePooling2D,UpSampling2D,Concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2,l1_l2,l1
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Add
from keras.initializers import RandomNormal
from keras.activations import relu
from keras import layers
from Models.layers import dense_c2f_gl, c2f_map, c2f_map_const_init, locally_connected_1d
import keras.constraints
import tensorflow as tf

MEAN_PIXELS = [123.68, 116.779, 103.939]

def subtract_mean(x):
    import tensorflow as tf
    mean = tf.constant(MEAN_PIXELS,shape=[1,1,1,3],dtype=tf.float32)
    return tf.subtract(x,mean)

def subtract_mean_3d(x):
    import tensorflow as tf
    mean = tf.constant(MEAN_PIXELS,shape=[1,1,1,1,3],dtype=tf.float32)
    return tf.subtract(x,mean)

def add_mean(x):
    import tensorflow as tf
    mean = tf.constant(MEAN_PIXELS,shape=[1,1,1,3],dtype=tf.float32)
    return tf.add(x,mean)


class encoder_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.resolution = 112
        self.conv_l1_reg = 1e-5
        self.conv_l2_reg = 0.001
        self.fc_reg_l1 = 20#1e-6 #20
        self.fc_reg_gl = 800#1.5e-4 #800
        self.fc_reg_gl_n = 0.5
        self.conv_ch = 32
        self.num_conv_layers = 2
        self.conv1_stride = 2
        self.conv_last_dim  =14
        self.drop_out = 0.5
        self.num_fmris_per_segment = 3
        self.num_frames_per_segment = 15
        self.ker_size = 3

        #seperable model
        self.c2f_l1 = 5e-6
        self.c2f_gl = 1e-5
        self.lc_l1 = 5e-6
        self.lc_l1_out = 5e-2
        self.patch = 3



def encoder_ml_seperable(param,vgg_loss,ch_mult = 1,name ='encoder'):
    in_img = Input((param.resolution, param.resolution, 3))

    x = Lambda(lambda img: img[:, :, :, ::-1] * 255.0)(in_img)
    x = Lambda(subtract_mean)(x)


    map28 = c2f_map_const_init(units=param.num_voxels, l1=param.c2f_l1, gl=param.c2f_gl)#, constraint= keras.constraints.NonNeg())
    map14 = c2f_map_const_init(units=param.num_voxels, l1=param.c2f_l1, gl=param.c2f_gl)#, constraint= keras.constraints.NonNeg())

    x1 = vgg_loss.layer_embed['block1_conv2'](x)
    conv1 = BatchNormalization(axis=-1)(x1)
    conv1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    conv1 = Conv2D(int(param.conv_ch*ch_mult), (3, 3), padding='same', kernel_initializer="glorot_normal", activation='relu',
                            kernel_regularizer=l1(param.conv_l1_reg), strides=(2, 2))(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Lambda(lambda x: tf.extract_image_patches(images=x, ksizes=[1, param.patch, param.patch, 1], strides=[1, 1, 1, 1],
                                                  rates=[1, 1, 1, 1], padding='VALID'))(conv1)
    conv1 = map28(conv1)
    conv1 = Dropout(param.drop_out)(conv1)
    out1 = locally_connected_1d(l1=param.lc_l1)(conv1)


    x2 = vgg_loss.layer_embed['block2_conv2'](x)
    conv2 = BatchNormalization(axis=-1)(x2)
    conv2 = Conv2D(int(param.conv_ch*ch_mult), (3, 3), padding='same', kernel_initializer="glorot_normal", activation='relu',
                            kernel_regularizer=l1(param.conv_l1_reg), strides=(2, 2))(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Lambda(lambda x: tf.extract_image_patches(images=x, ksizes=[1, param.patch, param.patch, 1], strides=[1, 1, 1, 1],
                                                  rates=[1, 1, 1, 1], padding='VALID'))(conv2)
    conv2 = map28(conv2)
    conv2 = Dropout(param.drop_out)(conv2)
    out2 = locally_connected_1d(l1=param.lc_l1)(conv2)


    x3 = vgg_loss.layer_embed['block3_conv3'](x)
    conv3 = BatchNormalization(axis=-1)(x3)
    conv3 = Conv2D(int(param.conv_ch*ch_mult), (3, 3), padding='same', kernel_initializer="glorot_normal", activation='relu',
                   kernel_regularizer=l1(param.conv_l1_reg), strides=(1, 1))(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Lambda(lambda x: tf.extract_image_patches(images=x, ksizes=[1, param.patch, param.patch, 1], strides=[1, 1, 1, 1],
                                                  rates=[1, 1, 1, 1], padding='VALID'))(conv3)
    conv3 = map28(conv3)
    conv3 = Dropout(param.drop_out)(conv3)
    out3 = locally_connected_1d(l1=param.lc_l1)(conv3)


    x4 = vgg_loss.layer_embed['block4_conv3'](x)
    conv4 = BatchNormalization(axis=-1)(x4)
    conv4 = Conv2D(int(param.conv_ch*ch_mult), (1, 1), padding='same', kernel_initializer="glorot_normal", activation='relu',
                            kernel_regularizer=l1(param.conv_l1_reg), strides=(1, 1))(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Lambda(lambda x: tf.extract_image_patches(images=x, ksizes=[1, param.patch, param.patch, 1], strides=[1, 1, 1, 1],
                                                  rates=[1, 1, 1, 1], padding='VALID'))(conv4)
    conv4 = map14(conv4)
    conv4 = Dropout(param.drop_out)(conv4)
    out4 = locally_connected_1d(l1=param.lc_l1)(conv4)


    out1 = Lambda(lambda x:tf.keras.backend.expand_dims(x,axis=-1))(out1)
    out2 = Lambda(lambda x:tf.keras.backend.expand_dims(x,axis=-1))(out2)
    out3 = Lambda(lambda x:tf.keras.backend.expand_dims(x,axis=-1))(out3)
    out4 = Lambda(lambda x:tf.keras.backend.expand_dims(x,axis=-1))(out4)
    out_concat = Concatenate(axis=-1)([out1, out2,out3,out4])#,out4,out5])#
    out = locally_connected_1d(l1=param.lc_l1_out, kernel_initializer ="ones", add_bias = False, constraint= keras.constraints.NonNeg())(out_concat)


    model = Model(inputs=in_img, outputs=out,name =name)
    return model
