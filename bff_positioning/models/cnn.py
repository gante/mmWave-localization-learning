""" Python class depicting a Convolutional Neural Network (CNN).
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .base_model import BaseModel
from .layer_functions import add_conv_layer, add_fc_layer


class CNN(BaseModel):
    """Convolutional Neural Network class

    :param model_settings: a dictionary containing the model settings
    """

    def __init__(self, model_settings):
        # Initializes the BaseModel
        super().__init__(model_settings=model_settings)

        # Instanciates CNN-specific variables
        self.input_reshape = model_settings["input_reshape"] \
            if "input_reshape" in model_settings else None
        self.conv_layers = model_settings["conv_layers"]
        self.conv_filters = model_settings["conv_filters"]
        self.conv_filter_size = model_settings["conv_filter_size"]
        self.conv_maxpool = model_settings["conv_maxpool"]
        self._check_conv_settings()

    # ---------------------------------------------------------------------------------------------
    # Model interface functions
    def set_graph(self):
        """ Sets the TF graph and initializes the session
        """
        # Sets: learning_rate_var, dropout, model_input, model_target, is_learning
        self._set_graph_io()

        # Adds the convolutional layers
        conv_output = None
        for conv_layer_idx in range(self.conv_layers):
            conv_output = add_conv_layer(
                self.conv_filter_size[conv_layer_idx],
                self.conv_filters[conv_layer_idx],
                self.conv_maxpool[conv_layer_idx],
                conv_output if conv_output is not None else self.model_input,
            )

        # Reshapes last convolutional output to a flat layer
        conv_elements = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]
        conv_output_flat = tf.reshape(conv_output, [-1, conv_elements])

        # Adds fully connected layers
        fcn_output = None
        for layer_idx in range(self.fc_layers):
            fcn_output = add_fc_layer(
                fcn_output if fcn_output is not None else conv_output_flat,
                self.fc_neurons,
                # last FC can't have dropout, otherwise the network will struggle to learn
                # (inputs to the output layer set to 0)
                self.dropout_var if layer_idx + 1 < self.fc_layers else 0.0
            )

        # Adds the output layer, storing the train step
        if self.output_type == "classification":
            self.train_step = self._add_classification_output(fcn_output)
        else:   # (regression)
            self.train_step = self._add_regression_output(fcn_output)

        # Sets: saver, session; Initializes TF variables
        self._prepare_model_for_training()

    def train_epoch(self, X, Y):
        """ Trains the model for an epoch. Uses the default training function
        (see BaseModel._train_epoch())

        :param X: numpy array with the features
        :param Y: numpy array with the labels
        """
        self._train_epoch(X, Y)

    def epoch_end(self, X=None, Y=None):
        """ Performs end of epoch operations, such as decaying the learning rate. Some
        operations require a validation set.

        :param X: numpy array with the validation features, defaults to None
        :param Y: numpy array with the validation labels, defaults to None
        """
        keep_training, val_score = self._epoch_end(X, Y)
        return keep_training, val_score

    def predict(self, X, validation=False):
        """ Returns the predictions on the given data

        :param X: numpy array with the features
        :param validation: boolean indicating whether these predictions have validation purposes
        :return: an numpy array with the predictions
        """
        return self._predict(X, validation=validation)

    def save(self, model_name="cnn"):
        """ Stores all model data inside the specified folder, given the model name

        :param model_name: the name of the model
        """
        self._save(model_name=model_name)

    def load(self, model_name="cnn"):
        """ Loads all model data from the specified folder, given the model name

        :param model_name: the name of the model
        """
        self._load(model_name=model_name)

    def close(self):
        """ Cleans up the session and any left over data
        """
        self._close_session()

    # ---------------------------------------------------------------------------------------------
    # Non-interface functions: misc
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
        assert len(self.input_shape) == 3 or len(self.input_reshape) == 3, "For a CNN network, "\
            "the input shape (or reshape) should have 3 dimentions (got {})".format(
            len(self.input_shape))
