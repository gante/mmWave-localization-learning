""" Python module with basic layer-related functions, to be reused in DL multiple architectures
"""

import numpy as np
import tensorflow as tf


# -------------------------------------------------------------------------------------------------
# Generalist functions
def weight_variable(shape, std=None):
    """ Function to initialize TF weights as random variables

    :param shape: the shape of the weight variable
    :param std: gaussian standard deviation for the initialization. If not passed,
        uses Xavier init
    """

    def _get_fans(shape):
        """Xavier initialization auxiliary function"""
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out

    #if the std is not explicitly added, uses Xavier init
    if std is None:
        fan_in, fan_out = _get_fans(shape)
        #TODO: try removing fan_out [https://arxiv.org/pdf/1502.01852.pdf] <----
        std = np.sqrt(2. / (fan_in + fan_out))
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)


def bias_variable(shape, bias=0.01):
    """Function to initialize the bias. For ReLUs, it MUST be > 0.0

    :param shape: the shape of the bias variable
    :param val: the value of the bias variable
    """
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial)


def batch_norm(x, phase):
    """Batch normalization wrapper"""
    return tf.layers.batch_normalization(x, center=True, scale=True, training=phase)


def add_linear_layer(input_data, n_dims, bias=0.01):
    """ Adds a linear layer to the input, returning its output. This is used as a basic
    building block for other layers

    :param input_data: TF tensor with this layer's input
    :param n_dims: Number of output dimentions
    :param bias: This layer's bias
    :returns: TF tensor with this layer's output
    """
    assert len(input_data.shape) == 2, "You must flatten your input before this layer!"
    input_length = input_data.shape[1]
    w = weight_variable([input_length, n_dims])
    b = bias_variable([n_dims], bias=bias)
    return tf.matmul(input_data, w) + b


def add_fc_layer(input_data, neurons, keep_prob):
    """ Adds a fully connected layer to the input, with dropout, returning its output

    :param input_data: TF tensor with this layer's input
    :param neurons: number this layer's neurons
    :param keep_prob: TF variable with (1 - dropout) probability
    :returns: TF tensor with this layer's output
    """
    h = tf.nn.relu(add_linear_layer(input_data, neurons))
    return tf.nn.dropout(h, keep_prob)


# -------------------------------------------------------------------------------------------------
# CNN functions
def conv2d(x, W):
    """2D convolution layer wrapper"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(data_in, x, y):
    """Maxpool layer wrapper"""
    return tf.nn.max_pool(data_in, ksize=[1, x, y, 1], strides=[1, x, y, 1], padding='SAME')


def add_conv_layer(filter_shape, n_filters, max_pool_shape, input_data):
    """ Adds a convolution layer with pooling, and returns its output

    :param filter_shape: filter dimentions
    :param n_filters: number of filters in the layer
    :param max_pool_shape: max pool dimentions
    :param input_data: TF tensor with this layer's input
    :returns: TF tensor with this layer's output
    """
    assert len(input_data.shape) == 4, "An input data tensor with 4 dimentions was expected"
    assert len(filter_shape) == 2, "Only 2D convolutions are supported (so far :D)"
    assert len(max_pool_shape) == 2, "Only 2D maxpool is supported (so far :D)"
    prev_layer_channels = input_data.shape[-1]
    w_conv = weight_variable([filter_shape[0], filter_shape[1], prev_layer_channels, n_filters])
    b_conv = bias_variable([n_filters])
    h_conv = tf.nn.relu(conv2d(input_data, w_conv) + b_conv)
    h_pool = max_pool(h_conv, max_pool_shape[0], max_pool_shape[1])
    return h_pool
