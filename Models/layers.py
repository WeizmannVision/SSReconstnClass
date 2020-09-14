from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.regularizers import Regularizer
from keras.regularizers import l2,l1_l2,l1
import keras
from Models.regularizers import *
def list_prod(l):
    prod = 1
    for e in l:
        prod*=e
    return prod



class SwitchLayer(Layer):
    def __init__(self, **kwargs):
        super(SwitchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SwitchLayer, self).build(input_shape)
        self.trainable = False


    def call(self, inputs):
        return K.switch(tf.constant(1), inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class temporal_fc(Layer):
    """preform dense operation only on the temporal dimension
       This operation removes the temporal dimension
       The operation is performed on each channel separately,
       producing a number of outputs for each channel and combines all the temporal information
       Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2  regularization factor.
        ch_mult: channel multiplier, number of output channels produced from each input channel
    """
    def __init__(self, ch_mult=4,l1=0.0,l2=0.0, **kwargs):
        self.ch_mult  = ch_mult
        self.l1 = l1
        self.l2 = l2
        super(temporal_fc, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='temporal_fc_kernel',
                                      shape=(input_shape[1],1 ,input_shape[4],self.ch_mult),
                                      regularizer=l1_l2(l1=self.l1, l2=self.l2),
                                      initializer="glorot_normal",
                                      trainable=True)

        self.bias = self.add_weight(name='temporal_fc_bias',shape=(input_shape[4]*self.ch_mult,),initializer="glorot_normal",trainable=True)

        self.s = input_shape
        super(temporal_fc, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        input_shape = self.s
        x = tf.reshape(x, shape = (-1,input_shape[1],input_shape[2]*input_shape[3],input_shape[4]))
        x = tf.nn.depthwise_conv2d(x,self.kernel,strides= [1,1,1,1] ,padding='VALID')
        # after operation second dim of x should have size 1
        x = tf.reshape(x, shape = (-1,input_shape[2],input_shape[3],input_shape[4]*self.ch_mult))
        x = K.bias_add(x,self.bias)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2],input_shape[3],input_shape[4]*self.ch_mult)



class temporal_shared_fc(Layer):
    """
       Same as temporal_fc, apart from sharing filters between channels
       Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2  regularization factor.
        ch_mult: channel multiplier, number of output channels produced from each input channel
        temporal_dim: keep temporal dimension, i.e. out put a 5D tensor , else merge the temporal dimension with the channels
    """



    def __init__(self, ch_mult=4,l1=0.0,l2=0.0,temporal_dim=0, **kwargs):
        self.ch_mult  = ch_mult
        self.l1 = l1
        self.l2 = l2
        self.temporal_dim = temporal_dim
        super(temporal_shared_fc, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='temporal_shared_fc_kernel',
                                      shape=(input_shape[1],1,1 ,1,self.ch_mult),
                                      regularizer=l1_l2(l1=self.l1, l2=self.l2),
                                      initializer="glorot_normal",
                                      trainable=True)

        self.bias = self.add_weight(name='temporal_shared_fc_bias',shape=(self.ch_mult,),initializer="glorot_normal",trainable=True)
        self.s = input_shape
        super(temporal_shared_fc, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        input_shape = self.s

        x = tf.reshape(x,[-1,input_shape[1],input_shape[2]*input_shape[3],input_shape[4],1])
        x = tf.nn.conv3d(x,self.kernel ,padding='VALID',strides=[1,1,1,1,1])
        x = K.bias_add(x, self.bias)
        if(self.temporal_dim):
            x = tf.reshape(x, [-1, input_shape[2] , input_shape[3], input_shape[4], self.ch_mult])
            x = tf.transpose(x,perm=[0,4,1,2,3])
        else:
            x = tf.reshape(x, [-1, input_shape[2] , input_shape[3], input_shape[4]* self.ch_mult])
        return x

    def compute_output_shape(self, input_shape):
        if(self.temporal_dim):
            return (input_shape[0], self.ch_mult, input_shape[2], input_shape[3], input_shape[4])
        else:
            return (input_shape[0], input_shape[2], input_shape[3], input_shape[4] * self.ch_mult)



class dense_c2f_gl(Layer):

    def __init__(self, units=1024,l1=0.1,l2=0.1,n_reg=0,W =None,kernel_init = "glorot_normal", **kwargs):
        self.units = units
        self.l1 = l1
        self.l2 = l2
        self.n_reg = n_reg
        self.W =W
        self.kernel_init =kernel_init
        super(dense_c2f_gl, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:])+[self.units]
        ##GroupLasso(l1=self.l1,l2=self.l2,n_reg= self.n_reg,W =self.W),
        self.kernel = self.add_weight(name='dense_c2f_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso(l1=self.l1,l2=self.l2,n_reg= self.n_reg,W =self.W) ,

                                      initializer=self.kernel_init,
                                      trainable=True)
        self.bias = self.add_weight(name='dense_c2f_gl_bias', shape=(self.units,),initializer="glorot_normal",trainable=True)

        super(dense_c2f_gl, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        input_shape = self.s
        size =  list_prod(input_shape[1:])
        x = tf.reshape(x, [-1, size])
        w = tf.reshape(self.kernel,[size ,self.units])
        output = K.dot(x, w)

        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
        #return tuple(input_shape[0],self.units)






class dense_c2c_gl(Layer):

    def __init__(self, out=[4,4,4],l1=0.1,gl=0.1,n_reg= 0.0, **kwargs):
        self.units = 1
        self.l1 = l1
        self.gl = gl
        self.n_reg = n_reg
        self.out = out
        super(dense_c2c_gl, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:])+self.out
        self.kernel = self.add_weight(name='dense_c2c_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso2D(l1=self.l1,l2=self.gl,n_reg=self.n_reg),
                                      initializer="glorot_normal",
                                      trainable=True)
        self.bias = self.add_weight(name='dense_c2c_gl_bias', shape=(self.out),initializer="glorot_normal",trainable=True)
        super(dense_c2c_gl, self).build(input_shape)


    def call(self, x):
        input_shape = self.s
        size = list_prod(input_shape[1:])
        x = tf.reshape(x, [-1, size])
        w = tf.reshape(self.kernel, [size, -1])
        output = K.dot(x, w)
        output = tf.reshape(output,[-1]+self.out)
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out[0], self.out[1], self.out[2])


class dense_c2c(Layer):

    def __init__(self, out=[4,4],l1=0.1,gl=0.1, **kwargs):
        self.units = 1
        self.l1 = l1
        self.gl = gl
        self.out = out
        super(dense_c2c, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:-1])+self.out
        self.kernel = self.add_weight(name='dense_c2c_kernel',
                                      shape=shape,
                                      regularizer= GroupLassoMap2D(self.l1,self.gl),
                                      initializer="glorot_normal",
                                      trainable=True)
        self.bias = self.add_weight(name='dense_c2c_bias', shape=(self.out+[self.s[3]]),initializer="glorot_normal",trainable=True)
        super(dense_c2c, self).build(input_shape)


    def call(self, x):
        input_shape = self.s
        #size = list_prod(input_shape[1:])
        #print(x.shape)
        x = tf.reshape(x, [-1, input_shape[1]*input_shape[2],input_shape[3]])
        #print(x.shape)

        x = tf.transpose(x, [0,2,1])
        #print(x.shape)


        w = tf.reshape(self.kernel, [input_shape[1]*input_shape[2],self.out[0]*self.out[1]])
        output = K.dot(x, w)
        output = tf.transpose(output, [0, 2, 1])
        output = tf.reshape(output,[-1]+self.out+[input_shape[3]])
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out[0], self.out[1],input_shape[3])


class dense_c22c3_gl(Layer):

    def __init__(self, out=[4,4,4,4],l1=0.1,l2=0.1,n_reg=0, **kwargs):
        self.out = out
        self.l1 = l1
        self.l2 = l2
        self.n_reg = n_reg
        #self.W =W
        super(dense_c22c3_gl, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:])+self.out
        self.kernel = self.add_weight(name='dense_c2f_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso3D(l1=self.l1,l2=self.l2,n_reg= self.n_reg),
                                      initializer="glorot_normal",
                                      trainable=True)
        #self.bias = self.add_weight(name='dense_c2f_gl_bias', shape=(self.out,),initializer="glorot_normal",trainable=True)

        super(dense_c22c3_gl, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        input_shape = self.s
        size =  list_prod(input_shape[1:])
        x = tf.reshape(x, [-1, size])
        w = tf.reshape(self.kernel,[size ,list_prod(self.out)])
        output = K.dot(x, w)

        output = tf.reshape(output,[-1]+self.out)
        #output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out)


class dense_f2c_gl(Layer):

    def __init__(self, out=[4,4,4],l1=0.1,l2=0.1,n_reg= 0.0,W=None,kernel_init = "glorot_normal", **kwargs):
        self.units = 1
        self.l1 = l1
        self.l2 = l2
        self.n_reg = n_reg
        self.out = out
        self.W = W
        self.kernel_init = kernel_init
        super(dense_f2c_gl, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = self.out+list(input_shape[1:])
        self.kernel = self.add_weight(name='dense_f2c_gl_kernel',
                                      shape=shape,
                                      regularizer= GroupLasso(l1=self.l1,l2=self.l2,n_reg=self.n_reg,W =self.W),
                                      initializer=self.kernel_init,
                                      trainable=True)
        self.bias = self.add_weight(name='dense_f2c_gl_bias', shape=(self.out),initializer="glorot_normal",trainable=True)
        super(dense_f2c_gl, self).build(input_shape)


    def call(self, x):
        input_shape = self.s
        len =  input_shape[1]
        w = tf.transpose(self.kernel,perm=[3,0,1,2])
        w = tf.reshape(w,[len,-1])
        output = K.dot(x, w)
        output = tf.reshape(output,[-1]+self.out)
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]]+ self.out)


class dense_f2c_gl_multi(Layer):

    def __init__(self, out=[4,4,4],l1=0.1,l2=0.1,n_reg= 0.0,num_sub = 5, **kwargs):
        self.units = 1
        self.l1 = l1
        self.l2 = l2
        self.n_reg = n_reg
        self.out = out
        self.num_sub = num_sub
        super(dense_f2c_gl_multi, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape[0]
        shape = self.out+list(self.s[1:])
        self.kernel = []
        self.bias = []
        for i in range(self.num_sub):
            self.kernel.append(self.add_weight(name='dense_f2c_gl_kernel',
                                          shape=shape,
                                          regularizer= GroupLasso(l1=self.l1,l2=self.l2,n_reg=self.n_reg),
                                          initializer="glorot_normal",
                                          trainable=True))
            self.bias.append(self.add_weight(name='dense_f2c_gl_bias', shape=(self.out),initializer="glorot_normal",trainable=True))

        self.kernel = tf.stack(self.kernel)
        self.bias = tf.stack(self.bias)


        super(dense_f2c_gl_multi, self).build(input_shape)


    def call(self, x):
        ind = x[1]
        x = x[0]
        input_shape = self.s
        len =  input_shape[1]
        ind_onehot = tf.one_hot(tf.cast(ind,tf.int32),self.num_sub)[0]
        kernel =  tf.squeeze(tf.tensordot(ind_onehot,self.kernel,axes = [[1],[0]]))

        bias = tf.squeeze(tf.tensordot(ind_onehot,self.bias,axes = [[1],[0]]))


        w = tf.transpose(kernel,perm=[3,0,1,2])
        w = tf.reshape(w,[len,-1])
        output = K.dot(x, w)
        output = tf.reshape(output,[-1]+self.out)
        output = K.bias_add(output, bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.out)


class c2f_map(Layer):

    def __init__(self, units=4,l1=0.1,gl=0.1, **kwargs):
        self.l1 = l1
        self.gl = gl
        self.units = units
        super(c2f_map, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = list(input_shape[1:-1])+[self.units]
        self.kernel = self.add_weight(name='dense_c2f_map_kernel',
                                      shape=shape,
                                      regularizer= GroupLassoMap(self.l1,self.gl),
                                      initializer="glorot_normal",
                                      trainable=True)
        #self.bias = self.add_weight(name='dense_c2f_map_bias', shape=([self.units,self.s[3]]),initializer="glorot_normal",trainable=True)
        super(c2f_map, self).build(input_shape)


    def call(self, x):
        input_shape = self.s

        x = tf.reshape(x, [-1, input_shape[1]*input_shape[2],input_shape[3]])

        x = tf.transpose(x, [0,2,1])


        w = tf.reshape(self.kernel, [input_shape[1]*input_shape[2],self.units])
        output = K.dot(x, w)
        output = tf.transpose(output, [0, 2, 1])
        #output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units,input_shape[3])


class c2f_map_const_init(Layer):

    def __init__(self, units=4,l1=0.1,gl=0.1, constraint =None, **kwargs):
        self.l1 = l1
        self.gl = gl
        self.units = units
        self.constraint = constraint
        super(c2f_map_const_init, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        len = input_shape[1]
        init = 2.0/(((len-2)*(len-2))+self.units)
        shape = list(input_shape[1:-1])+[self.units]
        self.kernel = self.add_weight(name='dense_c2f_map_kernel',
                                      shape=shape,
                                      regularizer= GroupLassoMap(self.l1,self.gl),
                                      initializer= keras.initializers.Constant(init),
                                      constraint=self.constraint,
                                      trainable=True)
        super(c2f_map_const_init, self).build(input_shape)


    def call(self, x):
        input_shape = self.s

        x = tf.reshape(x, [-1, input_shape[1]*input_shape[2],input_shape[3]])

        x = tf.transpose(x, [0,2,1])

        w = tf.reshape(self.kernel, [input_shape[1]*input_shape[2],self.units])
        output = K.dot(x, w)
        output = tf.transpose(output, [0, 2, 1])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units,input_shape[3])


class f2c_map_const_init(Layer):

    def __init__(self,len=4,l1=0.1,gl=0.1, constraint =None, **kwargs):
        self.l1 = l1
        self.gl = gl
        self.len = len
        self.constraint = constraint
        super(f2c_map_const_init, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        len = self.len
        units = input_shape[1]
        init = 2.0/(input_shape[-1]+units)
        shape = [units,len,len]
        self.kernel = self.add_weight(name='dense_c2f_map_kernel',
                                      shape=shape,
                                      regularizer= GroupLassoMap(self.l1,self.gl),
                                      initializer= keras.initializers.Constant(init),
                                      constraint=self.constraint,
                                      trainable=True)
        super(f2c_map_const_init, self).build(input_shape)


    def call(self, x):
        input_shape = self.s

        w = self.kernel
        #print(w.shape,x.shape)

        output = tf.tensordot(x, w,axes = [[1],[0]])
        #print(output.shape)
        output = tf.transpose(output, [0, 2 , 3 , 1])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.len, self.len,  input_shape[-1])

import keras.regularizers as reg
class locally_connected_1d(Layer):

    def __init__(self, out=1,l1=0.1, kernel_initializer = "glorot_normal"
,add_bias = True , constraint =None,single_dim = None,**kwargs):
        self.l1 = l1
        self.out = out
        self.kernel_initializer = kernel_initializer
        self.add_bias = add_bias
        self.constraint = constraint
        self.single_dim = single_dim
        super(locally_connected_1d, self).__init__( **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = [1]+list(input_shape[1:])
        if(self.single_dim is not None):
            shape[self.single_dim] =1

        self.kernel = self.add_weight(name='locally_connected_1d_kernel',
                                      shape=shape,
                                      regularizer= reg.l1(self.l1),
                                      initializer=self.kernel_initializer,
                                      constraint= self.constraint,
                                      trainable=True)
        if(self.add_bias):
            self.bias = self.add_weight(name='locally_connected_1d_bias', shape=([self.s[1]]),initializer="glorot_normal",trainable=True)
        super(locally_connected_1d, self).build(input_shape)


    def call(self, x):
        output = tf.multiply(x, self.kernel)
        output = K.sum(output,axis=-1)
        if (self.add_bias):
            output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0:-1]