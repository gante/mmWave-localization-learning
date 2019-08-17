""" Python class depicting a Convolutional Neural Network (CNN).
"""
#pylint: disable=unsubscriptable-object
#pylint: disable=attribute-defined-outside-init

import tensorflow as tf

from .base_model import ModelInterface, BASE_SETTINGS
from .layer_functions import add_conv_layer, add_fc_layer, add_linear_layer

ACCEPTED_SETTINGS = BASE_SETTINGS + [
    "conv_layers",
    "conv_filters",
    "conv_filter_size",
    "conv_maxpool",
]


class CNN(ModelInterface):
    """Convolutional Neural Network class

    :param model_settings: a dictionary containing the model settings
    """

    def __init__(self, model_settings):
        super().__init__(model_settings=model_settings)
        self._check_settings_names(ACCEPTED_SETTINGS)
        self._instatiate_cnn_variables()
        self._set_settings()
        self._check_conv_settings()
        self._set_gpu()

    def _instatiate_cnn_variables(self):
        """ Instatiates a bunch of model-specific variables (moved out of init for readibility)
        """
        self.conv_layers = None
        self.conv_filters = None
        self.conv_filter_size = None
        self.conv_maxpool = None

    def _check_conv_settings(self):
        """ Checks if the CNN's settings have the expected shape
        """
        # Checks conv layers
        for hyperparam in ("conv_filters", "conv_filter_size", "conv_maxpool"):
            assert self.conv_layers == len(getattr(self, hyperparam)), "The hyperparameter "\
                "{} does not has the expected dimension ({})".format(hyperparam, self.conv_layers)
        for hyperparam in ("conv_filter_size", "conv_maxpool"):
            for layer in range(self.conv_layers):
                assert len(getattr(self, hyperparam)[layer]) == 2, "Currently, only 2D "\
                    "convolutions are supported. (Check the {}-th layer on {})".format(layer,
                    hyperparam)
        # Checks other settings
        assert len(self.input_shape) == 3, "For a CNN network, the input shape should have 3 "\
            "dimentions (got {})".format(len(self.input_shape))

    def set_graph(self):
        """ Sets the TF graph
        """
        # Sets: learning_rate_var, keep_prob, model_input, model_target
        self._set_graph_io()

        # Adds the convolutional layers
        conv_output = None
        for layer_idx in range(self.conv_layers):
            conv_output = add_conv_layer(
                self.conv_filter_size[layer_idx],
                self.conv_filters[layer_idx],
                self.conv_maxpool[layer_idx],
                conv_output if conv_output else self.model_input,
            )

        # Reshapes last convolutional output to a flat layer
        conv_elements = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]
        conv_output_flat = tf.reshape(conv_output, [-1, conv_elements])

        # Adds fully connected layers
        fcn_output = None
        for _ in range(self.fc_layers):
            fcn_output = add_fc_layer(
                fcn_output if fcn_output else conv_output_flat,
                self.fc_neurons,
                self.keep_prob
            )

        #-------------------------------------------------------------------- rehash as common functions!
        # Adds the output layer and the corresponding loss function
        if self.output_type == "classification":

            # Defines the logits and the softmax output
            logits = add_linear_layer(fcn_output, self.output_shape)
            tf.nn.softmax(logits, name=self.output_name)

            # Defines the loss function [mean(cross_entropy(target_value - obtained_softmax))]
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.model_target,
                logits=logits
            ))

            # Defines the optimizer (ADAM) and the train step
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var)
            self.train_step = optimizer.minimize(cross_entropy)

        else:   # (regression)
