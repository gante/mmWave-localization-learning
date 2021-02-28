""" Python module with basic layer-related functions, to be reused in DL multiple architectures
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# -------------------------------------------------------------------------------------------------
# Generalist model functions
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
    initial = tf.random.truncated_normal(shape, stddev=std)
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
    input_length = int(input_data.shape[1])
    w = weight_variable([input_length, n_dims])
    b = bias_variable([n_dims], bias=bias)
    return tf.matmul(input_data, w) + b


def add_fc_layer(input_data, neurons, dropout):
    """ Adds a fully connected layer to the input, with dropout, returning its output

    :param input_data: TF tensor with this layer's input
    :param neurons: number this layer's neurons
    :param dropout: TF variable with the dropout probability
    :returns: TF tensor with this layer's output
    """
    h = tf.nn.relu(add_linear_layer(input_data, neurons))
    return tf.nn.dropout(h, rate=dropout)


# -------------------------------------------------------------------------------------------------
# CNN functions
def conv2d(x, W):
    """2D convolution layer wrapper"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(data_in, x, y):
    """Maxpool layer wrapper"""
    return tf.nn.max_pool2d(data_in, ksize=[1, x, y, 1], strides=[1, x, y, 1], padding='SAME')


def add_conv_layer(filter_shape, n_filters, max_pool_shape, input_data):
    """ Adds a convolution layer with pooling, and returns its output

    :param filter_shape: filter dimentions
    :param n_filters: number of filters in the layer
    :param max_pool_shape: max pool dimentions
    :param input_data: TF tensor with this layer's input
    :returns: TF tensor with this layer's output
    """
    assert len(input_data.shape) == 4, "An input data tensor with 4 dimentions was expected. "\
        "(Got {} dimentions)".format(input_data.shape)
    assert len(filter_shape) == 2, "Only 2D convolutions are supported (so far :D)"
    assert len(max_pool_shape) == 2, "Only 2D maxpool is supported (so far :D)"
    prev_layer_channels = int(input_data.shape[-1])
    w_conv = weight_variable([filter_shape[0], filter_shape[1], prev_layer_channels, n_filters])
    b_conv = bias_variable([n_filters])
    h_conv = tf.nn.relu(conv2d(input_data, w_conv) + b_conv)
    h_pool = max_pool(h_conv, max_pool_shape[0], max_pool_shape[1])
    return h_pool

# -------------------------------------------------------------------------------------------------
# TCN functions
# Adapted from * and the code therein.
# * = https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7
class CausalConv1D(tf.compat.v1.layers.Conv1D):
    """1D convolution with padding"""
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)])
            * padding)
        return super(CausalConv1D, self).call(inputs)


class TemporalBlock(tf.compat.v1.layers.Layer):
    """ TCN's "layer", containing the residual network and the convolutions, with
    adjustable dilation.
    """
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate,
                 dropout=0.2, trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv2")
        self.down_sample = None


    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.compat.v1.layers.Dropout(self.dropout, [tf.constant(1),
            tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.compat.v1.layers.Dropout(self.dropout, [tf.constant(1),
            tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.compat.v1.layers.Dense(self.n_outputs, activation=None)
        self.built = True

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)


class TemporalConvNet(tf.compat.v1.layers.Layer):
    """TCN's main class. See the __init__ for more parameter information

    Key Input Parameters:
        num_channels = list containing the number of features for each TCN
            layer. Please note that the dilation size is always
            2**(current_layer_index)!
            (e.g. num_channels=[20,20,20] has 3 layers with 20 features
            each, with dilations = [1,2,4])
        kernel_size = length of the temporal filter
        dropout = dropout probability
    """
    def __init__(self, num_channels=None, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):

        assert num_channels is not None, "Please read the docstring above and"\
            " define the num_channels variable!"

        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for idx in range(num_levels):
            dilation_size = 2 ** idx
            out_channels = num_channels[idx]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1,
                              dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(idx))
            )

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs
