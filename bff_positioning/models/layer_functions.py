""" Python module with layer-related functions, to be reused in DL multiple architectures
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


def bias_variable(shape, val=0.01):
    """Function to initialize the bias. For ReLUs, it MUST be > 0.0

    :param shape: the shape of the bias variable
    :param val: the value of the bias variable
    """
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)


def batch_norm(x, phase):
    """Batch normalization wrapper"""
    return tf.layers.batch_normalization(x, center=True, scale=True, training=phase)


# -------------------------------------------------------------------------------------------------
# CNN functions
def conv2d(x, W):
    """2D convolution layer wrapper"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(data_in, x, y):
    """Maxpool layer wrapper"""
    return tf.nn.max_pool(data_in, ksize=[1, x, y, 1], strides=[1, x, y, 1], padding='SAME')
