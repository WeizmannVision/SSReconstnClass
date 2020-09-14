from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.regularizers import Regularizer
from keras.regularizers import l2,l1_l2,l1

class GroupLasso(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., l2=0.,n_reg=0,W = None):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.n_reg = K.cast_to_floatx(n_reg)
        self.W_is_not_none = W is not None
        if(W is not None):
            self.W = K.cast_to_floatx(W)

    def __call__(self, x):

        if (self.W_is_not_none):
            if(self.W.ndim<4):
                W = tf.reshape(self.W,[1,1,1,-1])
            else:
                W = self.W
            regularization = self.l1 * K.mean(tf.multiply(K.abs(x),W))
            #regularization = self.l1 * K.sum(tf.multiply(K.abs(x), W))
        else:
            regularization = self.l1 * K.mean(K.abs(x))
            #regularization = self.l1 * K.sum(K.abs(x))
        x_sq = K.square(x)

        # if (self.W_is_not_none):
        #     x_sq = tf.tensordot(x_sq,self.W,axes = [[3],[0]])


        if (self.n_reg > 0):
            x_sq_pad = tf.pad(x_sq, [[1, 1], [1, 1], [0, 0], [0, 0]], "SYMMETRIC")
            x_sq_avg_n = (x_sq_pad[:-2, 1:-1] + x_sq_pad[2:, 1:-1] + x_sq_pad[1:-1, :-2] + x_sq_pad[1:-1, 2:]) / 4
            x_sq = (x_sq + self.n_reg * x_sq_avg_n) / (1 + self.n_reg)
        #
        # if (self.W_is_not_none):
        #     regularization += self.l2 * K.mean(K.sqrt(K.mean(tf.multiply(x_sq,W), axis=-2)))
        # else:
        regularization += self.l2 * K.mean(K.sqrt(K.mean(x_sq, axis=-2)))  # assumes weight structure x,y,ch,voxel
        #regularization += self.l2 * K.sum(K.sqrt(K.sum(x_sq, axis=-2)))





        return regularization





    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


class GroupLasso2D(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., l2=0.,n_reg=0,W = None):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.n_reg = K.cast_to_floatx(n_reg)


    def __call__(self, x):


        regularization = self.l1 * K.sum(K.abs(x))


        x_sq = K.square(x)

        # if (self.W_is_not_none):
        #     x_sq = tf.tensordot(x_sq,self.W,axes = [[3],[0]])
        #print('x_sq',x_sq.shape)
        s = x.shape

        if (self.n_reg > 0):
            x_sq_pad_img = tf.reshape(x_sq,[s[0],s[1],-1])  ## padding for 6d tensors not implemented
            x_sq_pad_img = tf.pad(x_sq_pad_img, [[1, 1], [1, 1],[0,0]], "SYMMETRIC")
            x_sq_pad_img = tf.reshape(x_sq_pad_img,s)  ## padding for 6d tensors not implemented
            x_sq_avg_img_n = (x_sq_pad_img[:-2, 1:-1] + x_sq_pad_img[2:, 1:-1] + x_sq_pad_img[1:-1, :-2] + x_sq_pad_img[1:-1, 2:]) / 4

            x_sq_pad_img_vox = tf.reshape(x_sq, [s[0]*s[1]*s[2],s[3],s[4],s[5]])
            x_sq_pad_img_vox = tf.pad(x_sq_pad_img_vox, [ [0, 0], [1, 1],[1,1],[0,0]], "SYMMETRIC")
            x_sq_pad_img_vox = tf.reshape(x_sq_pad_img_vox, s)
            x_sq_avg_img_vox_n = (x_sq_pad_img_vox[:,:-2, 1:-1] + x_sq_pad_img_vox[:,2:, 1:-1] + x_sq_pad_img_vox[:,1:-1, :-2] + x_sq_pad_img_vox[:,1:-1, 2:]) / 4

            x_sq = (x_sq + self.n_reg * x_sq_avg_img_n+ self.n_reg * x_sq_avg_img_vox_n) / (1 + 2*self.n_reg)
        #
        # if (self.W_is_not_none):
        #     regularization += self.l2 * K.mean(K.sqrt(K.mean(tf.multiply(x_sq,W), axis=-2)))
        # else:
        regularization += self.l2 * K.sum(K.sqrt(K.sum(x_sq, axis=[2,-1])))  # assumes weight structure x,y,ch,voxel
        #regularization += self.l2 * K.sum(K.sqrt(K.sum(x_sq, axis=-2)))





        return regularization





    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


class GroupLassoMap2D(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., l2=0.,W = None):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)


    def __call__(self, x):


        regularization = self.l1 * K.sum(K.abs(x))


        x_sq = K.square(x)

        # if (self.W_is_not_none):
        #     x_sq = tf.tensordot(x_sq,self.W,axes = [[3],[0]])
        #print('x_sq',x_sq.shape)

        #x_sq_pad_img = tf.reshape(x_sq,[s[0],s[1],-1])  ## padding for 6d tensors not implemented
        x_sq_pad_img = tf.pad(x_sq, [[1, 1], [1, 1],[0,0],[0,0]], "SYMMETRIC")
        #x_sq_pad_img = tf.reshape(x_sq_pad_img,s)  ## padding for 6d tensors not implemented
        x_sq_avg_img_n = (x_sq_pad_img[:-2, 1:-1] + x_sq_pad_img[2:, 1:-1] + x_sq_pad_img[1:-1, :-2] + x_sq_pad_img[1:-1, 2:]) / 4

        #x_sq_pad_img_vox = tf.reshape(x_sq, [s[0]*s[1]*s[2],s[3],s[4],s[5]])
        x_sq_pad_img_vox = tf.pad(x_sq, [[0, 0], [0, 0], [1, 1],[1,1]], "SYMMETRIC")
        #x_sq_pad_img_vox = tf.reshape(x_sq_pad_img_vox, s)
        x_sq_avg_img_vox_n = (x_sq_pad_img_vox[:,:,:-2, 1:-1] + x_sq_pad_img_vox[:,:,2:, 1:-1] + x_sq_pad_img_vox[:,:,1:-1, :-2] + x_sq_pad_img_vox[:,:,1:-1, 2:]) / 4

        #x_sq = ( x_sq_avg_img_n + x_sq_avg_img_vox_n)

        #
        # if (self.W_is_not_none):
        #     regularization += self.l2 * K.mean(K.sqrt(K.mean(tf.multiply(x_sq,W), axis=-2)))
        # else:
        regularization += self.l2 * (K.sum(K.sqrt(x_sq_avg_img_n))+(K.sum(K.sqrt(x_sq_avg_img_vox_n)))) # assumes weight structure x,y,ch,voxel
        #regularization += self.l2 * K.sum(K.sqrt(K.sum(x_sq, axis=-2)))



        return regularization





    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


class GroupLasso3D(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., l2=0.,n_reg=0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.n_reg = K.cast_to_floatx(n_reg)


    def __call__(self, x):


        regularization = self.l1 * K.mean(K.abs(x))
        x_sq = K.square(x)

        # if (self.W_is_not_none):
        #     x_sq = tf.tensordot(x_sq,self.W,axes = [[3],[0]])

        if (self.n_reg > 0):
            s = tf.shape(x_sq)
            x_sq_pad = tf.reshape(x_sq,[s[0],s[1],-1])
            x_sq_pad = tf.pad(x_sq_pad, [[1, 1], [1, 1], [0, 0]], "SYMMETRIC")


            x_sq_pad = tf.reshape(x_sq_pad, [s[0]+2,s[1]+2,s[2],s[3],s[4],s[5],s[6]])
            x_sq_avg_n = (x_sq_pad[:-2, 1:-1] + x_sq_pad[2:, 1:-1] + x_sq_pad[1:-1, :-2] + x_sq_pad[1:-1, 2:]) / 4
            x_sq = (x_sq + self.n_reg * x_sq_avg_n) / (1 + self.n_reg)
        #
        # if (self.W_is_not_none):
        #     regularization += self.l2 * K.mean(K.sqrt(K.mean(tf.multiply(x_sq,W), axis=-2)))
        # else:
        regularization += self.l2 * K.mean(K.sqrt(K.mean(x_sq, axis=[2,-1])))  # assumes weight structure x,y,ch,voxel


        return regularization





    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}



class L1(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)


    def __call__(self, x):
        regularization = self.l1 * K.mean(K.abs(x))

        return regularization

    def get_config(self):
        return {'l1': float(self.l1)}



class L1L2(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0.,l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = l2


    def __call__(self, x):
        regularization = self.l1 * K.mean(K.abs(x))
        regularization += self.l2 *K.sqrt(K.mean(K.square(x)))

        return regularization

    def get_config(self):
        return {'l1': float(self.l1)}


class GroupLassoMap(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 group regularization factor.
        n_reg: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., l2=0.,W = None):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)


    def __call__(self, x):
        regularization = self.l1 * K.sum(K.abs(x))
        x_sq = K.square(x)
        x_sq_pad_img = tf.pad(x_sq, [[1, 1], [1, 1],[0,0]], "SYMMETRIC")
        x_sq_avg_img_n = (x_sq_pad_img[:-2, 1:-1] + x_sq_pad_img[2:, 1:-1] + x_sq_pad_img[1:-1, :-2] + x_sq_pad_img[1:-1, 2:]) / 4

        regularization += self.l2 * (K.sum(K.sqrt(x_sq_avg_img_n))) # assumes weight structure x,y,ch,voxel
        return regularization


    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}
