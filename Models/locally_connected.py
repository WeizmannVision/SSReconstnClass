from keras.engine.topology import Layer
from Models.regularizers import *

import keras.regularizers as reg

from tensorflow.python.keras.utils import conv_utils
from keras.initializers import VarianceScaling

class LocallyConnected2D_(Layer):
    def __init__(self,out_ch,kernel_regularizer = None ):
        super(LocallyConnected2D_, self).__init__()
        self.out_ch = out_ch
        self.kernel_regularizer = kernel_regularizer


    def build(self, input_shape):
        input_row, input_col = input_shape[1],input_shape[2]
        input_filter = input_shape[3]


        self.kernel_shape = (input_row, input_col, input_filter,
                             input_row, input_col, self.out_ch)

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                    initializer="glorot_normal",
                                    name='kernel',
                                    regularizer=self.kernel_regularizer)

        self.kernel_mask = get_locallyconnected_mask(
          input_shape=(input_row, input_col),
          kernel_shape=(1,1),
          strides=(1,1),
          padding="valid",
          data_format="channels_last"
        )


        self.bias = self.add_weight(shape=(input_row, input_col, self.out_ch),name='bias',initializer="glorot_normal",trainable=True)
        super(LocallyConnected2D_, self).build(input_shape)



    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_ch)


    def call(self, inputs):

        output = local_conv_matmul(inputs, self.kernel, self.kernel_mask,
                                 self.compute_output_shape(inputs.shape))

        print(inputs.shape)
        print(output.shape)
        print(self.bias.shape)
        output = K.bias_add(output, self.bias)

        return output


def get_locallyconnected_mask(input_shape,
                              kernel_shape,
                              strides,
                              padding,
                              data_format):
  """Return a mask representing connectivity of a locally-connected operation.
  This method returns a masking numpy array of 0s and 1s (of type `np.float32`)
  that, when element-wise multiplied with a fully-connected weight tensor, masks
  out the weights between disconnected input-output pairs and thus implements
  local connectivity through a sparse fully-connected weight tensor.
  Assume an unshared convolution with given parameters is applied to an input
  having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
  to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
  by layer parameters such as `strides`).
  This method returns a mask which can be broadcast-multiplied (element-wise)
  with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer between
  (N+1)-D activations (N spatial + 1 channel dimensions for input and output)
  to make it perform an unshared convolution with given `kernel_shape`,
  `strides`, `padding` and `data_format`.
  Arguments:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`
                 spatial shape of the input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel
                  / receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
    data_format: a string, `"channels_first"` or `"channels_last"`.
  Returns:
    a `np.float32`-type `np.ndarray` of shape
    `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
    if `data_format == `"channels_first"`, or
    `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
    if `data_format == "channels_last"`.
  Raises:
    ValueError: if `data_format` is neither `"channels_first"` nor
                `"channels_last"`.
  """
  mask = conv_utils.conv_kernel_mask(
      input_shape=input_shape,
      kernel_shape=kernel_shape,
      strides=strides,
      padding=padding
  )

  ndims = int(mask.ndim / 2)

  if data_format == 'channels_first':
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, -ndims - 1)

  elif data_format == 'channels_last':
    mask = np.expand_dims(mask, ndims)
    mask = np.expand_dims(mask, -1)

  else:
    raise ValueError('Unrecognized data_format: ' + str(data_format))

  return mask

def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
  """Apply N-D convolution with un-shared weights using a single matmul call.
  This method outputs `inputs . (kernel * kernel_mask)`
  (with `.` standing for matrix-multiply and `*` for element-wise multiply)
  and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
  hence perform the same operation as a convolution with un-shared
  (the remaining entries in `kernel`) weights. It also does the necessary
  reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.
  Arguments:
      inputs: (N+2)-D tensor with shape
          `(batch_size, channels_in, d_in1, ..., d_inN)`
          or
          `(batch_size, d_in1, ..., d_inN, channels_in)`.
      kernel: the unshared weights for N-D convolution,
          an (N+2)-D tensor of shape:
          `(d_in1, ..., d_inN, channels_in, d_out2, ..., d_outN, channels_out)`
          or
          `(channels_in, d_in1, ..., d_inN, channels_out, d_out2, ..., d_outN)`,
          with the ordering of channels and spatial dimensions matching
          that of the input.
          Each entry is the weight between a particular input and
          output location, similarly to a fully-connected weight matrix.
      kernel_mask: a float 0/1 mask tensor of shape:
           `(d_in1, ..., d_inN, 1, d_out2, ..., d_outN, 1)`
           or
           `(1, d_in1, ..., d_inN, 1, d_out2, ..., d_outN)`,
           with the ordering of singleton and spatial dimensions
           matching that of the input.
           Mask represents the connectivity pattern of the layer and is
           precomputed elsewhere based on layer parameters: stride,
           padding, and the receptive field shape.
      output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)`
          or
          `(batch_size, d_out1, ..., d_outN, channels_out)`,
          with the ordering of channels and spatial dimensions matching that of
          the input.
  Returns:
      Output (N+2)-D tensor with shape `output_shape`.
  """
  inputs_flat = K.reshape(inputs, (K.shape(inputs)[0], -1))

  kernel = kernel_mask * kernel
  kernel = make_2d(kernel, split_dim=K.ndim(kernel) // 2)

  output_flat = tf.sparse_matmul(inputs_flat, kernel, b_is_sparse=True)
  output = K.reshape(output_flat,
                     [K.shape(output_flat)[0],] + list(output_shape)[1:])
  return output


def make_2d(tensor, split_dim):
  """Reshapes an N-dimensional tensor into a 2D tensor.
  Dimensions before (excluding) and after (including) `split_dim` are grouped
  together.
  Arguments:
    tensor: a tensor of shape `(d0, ..., d(N-1))`.
    split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).
  Returns:
    Tensor of shape
    `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
  """
  shape = tensor.shape
  in_dims = shape[:split_dim]
  out_dims = shape[split_dim:]

  in_size = tf.math.reduce_prod(in_dims)
  out_size = tf.math.reduce_prod(out_dims)

  return tf.reshape(tensor, (in_size, out_size))




def make_2d(tensor, split_dim):
  """Reshapes an N-dimensional tensor into a 2D tensor.
  Dimensions before (excluding) and after (including) `split_dim` are grouped
  together.
  Arguments:
    tensor: a tensor of shape `(d0, ..., d(N-1))`.
    split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).
  Returns:
    Tensor of shape
    `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
  """
  shape = tensor.shape
  in_dims = shape[:split_dim]
  out_dims = shape[split_dim:]

  in_size = tf.math.reduce_prod(in_dims)
  out_size = tf.math.reduce_prod(out_dims)

  return tf.reshape(tensor, (in_size, out_size))



#,kernel_initializer=

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
        return input_shape[0:2]

class locally_connected_1d_ch(Layer):

    #"glorot_normal"
    def __init__(self, out=1, l1=0.1,
                 kernel_initializer = VarianceScaling(scale=10.0)
                 , add_bias=True, constraint=None, single_dim=None, **kwargs):  # "glorot_normal",
        self.l1 = l1
        self.out = out
        self.kernel_initializer = kernel_initializer
        self.add_bias = add_bias
        self.constraint = constraint
        self.single_dim = single_dim
        super(locally_connected_1d_ch, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s = input_shape
        shape = [1] + list(input_shape[1:])+[self.out]
        if (self.single_dim is not None):
            shape[self.single_dim] = 1

        self.kernel = self.add_weight(name='locally_connected_1d_kernel',
                                      shape=shape,
                                      regularizer=reg.l1(self.l1),
                                      initializer=self.kernel_initializer,
                                      constraint=self.constraint,
                                      trainable=True)
        if (self.add_bias):
            self.bias = self.add_weight(name='locally_connected_1d_bias', shape=([self.s[1],self.out]),
                                        initializer="glorot_normal", trainable=True)
        super(locally_connected_1d_ch, self).build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x,axis=-1)
        output = tf.multiply(x, self.kernel)
        output = K.sum(output, axis=-2)
        if (self.add_bias):
            output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0:2]+(self.out,)






    # mode = 'fan_out',
    # distribution = 'normal'



