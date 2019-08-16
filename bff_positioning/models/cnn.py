""" Python class depicting a Convolutional Neural Network (CNN).
"""
# pylint: disable=no-member

import tensorflow as tf

from .base_model import ModelInterface, BASE_SETTINGS

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
        self._check_conv_settings()
        self._set_gpu()

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
        # Sets: learning_rate_var, keep_prob, model_input, model_output
        self._set_graph_io()
