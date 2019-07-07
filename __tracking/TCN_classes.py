'''
TCN_classes.py

-> Defines the basic TCN block class

Adapted from * and the code therein.
* = https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7
'''

import tensorflow as tf

# -----------------------------------------------------------------------------
class CausalConv1D(tf.layers.Conv1D):
    '''1D convolution with padding'''
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


# -----------------------------------------------------------------------------
class TemporalBlock(tf.layers.Layer):
    '''
    TCN's "layer", containing the residual network and the convolutions, with
        adjustable dilation.
    '''
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
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1),
            tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1),
            tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
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


# -----------------------------------------------------------------------------
class TemporalConvNet(tf.layers.Layer):
    '''TCN's main class. See the __init__ for more parameter information'''
    def __init__(self, num_channels=None, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        '''
        Defines the full TCN.

        Key Input Parameters:
            num_channels = list containing the number of features for each TCN
                layer. Please note that the dilation size is always
                2**(current_layer_index)!
                (e.g. num_channels=[20,20,20] has 3 layers with 20 features
                each, with dilations = [1,2,4])
            kernel_size = length of the temporal filter
            dropout = dropout probability
        '''

        assert num_channels is not None, "Please read the docstring above and"\
            " define the num_channels variable!"

        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1,
                              dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs