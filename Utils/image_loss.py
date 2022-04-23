from keras.applications import vgg19
from keras.applications import vgg16
from keras.applications import resnet50
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input
from keras.losses import mean_squared_error,cosine_proximity,mean_absolute_error, kullback_leibler_divergence, categorical_crossentropy
from keras import backend as K
from keras.optimizers import SGD,Adam, RMSprop
import numpy as np
from keras.layers import Lambda
from scipy.misc import imresize
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input




img_len = 112

MEAN_PIXELS = [123.68, 116.779, 103.939]
shift_inv = 0
Tv_reg = 0.3

dir_ ='data/ImageNet_Files/images/'

img_norm = np.load(dir_+'img_norm.npz')

vgg_ch = {}
vgg_ch['block1_conv2'] = 64
vgg_ch['block2_conv2'] = 128
vgg_ch['block3_conv2'] = 256
vgg_ch['block4_conv2'] = 512
vgg_ch['block5_conv2'] = 512


scales = [0, 1, 1, 1, 1, 1]

class vgg_layer_loss():
    def __init__(self,img_len = 56 ,ignore_edge =0 , train_vgg = False):
        self.layer_embed = {}
        self.vgg_norm = {}
        self.ignore_edge = ignore_edge
        in_img = Input(shape=(img_len, img_len, 3))
        self.img_len = img_len
        x = Lambda(self.vgg_in)(in_img)
        model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_len, img_len, 3), input_tensor=x)
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        selectedLayers = [2, 3, 6, 10, 14, 18]
        selectedOutputs = [model.layers[i].output for i in selectedLayers]

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        for layer in model.layers:
            self.layer_embed[layer.name] = Model(inputs=in_img, outputs=outputs_dict[layer.name])

        self.lossModel = Model(model.inputs, selectedOutputs)

    def calc_norm_factors(self,imgs):
        for l in range(1,6):
            layer= 'block'+str(l)+'_conv2'
            embed = self.layer_embed[layer].predict(imgs,batch_size=64)
            self.vgg_norm[layer] = np.mean(np.abs(embed).reshape(-1,embed.shape[3]),axis=0).reshape(1,1,1,-1)


    def normalize(self,y):
        norm = K.sqrt(K.sum(K.square(y),axis=[3],keepdims=True))
        y_norm = tf.divide(y, norm + 1e-10)
        return y_norm


    def loss(self,y_true, y_pred):
        y_true_e = self.lossModel(y_true)
        y_pred_e = self.lossModel(y_pred)
        res = 0
        for i in range(len(y_true_e)):
            y_true_en = self.normalize(y_true_e[i])
            y_pred_en = self.normalize(y_pred_e[i])
            res += scales[i]*self.cos_dist(y_true_en,y_pred_en, i)
        return res/(np.sum(scales))


    def cos_dist(self,y_true,y_pred, layer):
        return (1 - K.expand_dims(K.mean(K.mean(K.sum(y_true*y_pred,axis=[3]),axis=[2]),axis=[1]),axis=-1))


    def vgg_in(self,x):
        x = tf.scalar_mul(255.0, x)
        mean = tf.constant(MEAN_PIXELS, shape=[1, 1, 1, 3], dtype=tf.float32)
        x = tf.subtract(x, mean)
        x = x[:, :, :, ::-1]
        return x



vgg_loss = vgg_layer_loss(ignore_edge =shift_inv,img_len = img_len)

def total_variation_loss(x):
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, : - 1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, : - 1, 1:, :])
    return K.mean(K.pow(a + b, 1.25))

def vgg_total_loss(y_true, y_pred):
    return vgg_loss.loss(y_true,y_pred)


def mae(y_true, y_pred):
    y_true_norm = tf.divide(y_true,img_norm['l1'])
    y_pred_norm = tf.divide(y_pred,img_norm['l1'])
    return K.expand_dims(K.mean(K.abs(y_true_norm-y_pred_norm),axis=[1,2,3]),axis=-1)

def mae_gray_scale(y_true, y_pred):
    y_true_gray = tf.image.rgb_to_grayscale(y_true)
    y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
    return 2*K.mean(K.abs(y_true_gray-y_pred_gray),axis=[1,2])

def mse(y_true, y_pred):
    y_true_norm = tf.divide(y_true, img_norm['l2'])
    y_pred_norm = tf.divide(y_pred, img_norm['l2'])
    return K.expand_dims(K.mean(K.square(y_true_norm-y_pred_norm),axis=[1,2,3]),axis=-1)



