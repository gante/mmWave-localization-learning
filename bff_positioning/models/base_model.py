""" Python module containing a class for basic interface of a ML model, as well as functions
shared by multiple model types.
"""
# pylint: disable=no-member

import os
import logging

import tensorflow as tf

# Settings common to all model types
BASE_SETTINGS = [
    # IO settings
    "input_name",
    "input_shape",
    "output_name",
    "output_type",
    "output_shape",

    # General hyperparams
    "batch_size",
    "batch_size_inference",
    "dropout",
    "epochs",
    "fc_layers",
    "fc_neurons",
    "learning_rate",
    "learning_rate_decay",
    "target_gpu",
]


class ModelInterface():
    """ This class defines a common model interface. The models defined within this module
    should implement all depicted functions, to abstract the details away from the main scripts

    :param model_settings: a dictionary containing the model settings. All dictionary key/value
        pairs will be set as class members, for further readibility
    """

    # To add in the future:
    # 1) Instead of "train_batch()" with an outer control loop, create a "train_generator()" that
    #   uses generators
    # 2) Similarly, crate a "predict_generator()"
    # 3) Add an MC dropout flag to the predict functions, to enable uncertainty prediction

    def __init__(self, model_settings):
        self.model_settings = model_settings
        for key, value in model_settings.items():
            setattr(self, key, value)

        # Instantiates other variables
        self.learning_rate_var = None
        self.keep_prob = None
        self.model_input = None
        self.model_output = None

    def _check_settings_names(self, accepted_settings):
        """Checks the all input settings are usable. If they are not, it is likely that
        there was some misplanning with the model configuration, and it should be re-checked

        :param accepted_settings: list of strings with the accepted settings for each model
        """
        error_str = "Unexpected settings were found. Please double check the settings file!"\
            "\nList of expected settings: {}\nList of obtained settings: {}"
        assert set(self.model_settings.keys()) == set(accepted_settings), error_str.format(
            accepted_settings, list(self.model_settings.keys()))

    def _set_gpu(self):
        """ If the option 'target_gpu' is defined, sets that GPU
        """
        if hasattr(self, "target_gpu"):
            logging.info("[Using GPU #%s]", self.target_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.target_gpu)

    def set_graph(self):
        """Prototype: setup(self)

        Given the settings, sets up the model graph for training
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'setup()' function."
        )

    def _set_graph_io(self):
        """ Auxiliary function to "set_graph()". Sets the graph input and trainable target,
        as well as some other basic variables common to all model types
        """
        # The current learning rate
        self.learning_rate_var = tf.placeholder(tf.float32, shape=[])
        # (1 - Dropout) probability
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.model_input = tf.placeholder(
            tf.float32,
            shape=[None] + self.input_shape,
            name=self.input_name,
        )
        if self.output_type == "regression":
            self.model_output = tf.placeholder(
                tf.float32,
                shape=[None] + self.output_shape,
                name=self.output_name
            )
        elif self.output_type == "classification":
            self.model_output = tf.placeholder(tf.int64, shape=[None], name =self.output_name)
        else:
            raise ValueError("Unknown 'output_type' ({}). Only 'classification' and 'regression' "
                "are accepted.".format(self.output_type))

    def train_batch(self, batch_x, batch_y):
        """Prototype: train_batch(self, batch_x, batch_y)

        Trains on a single batch. Requires an outer control loop, feeding the data.
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'train_batch()' function."
        )

    def epoch_end(self):
        """Prototype: epoch_end(self)

        Executes end of epoch logic (e.g. decay learning rate, try early stopping, ...)
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'epoch_end()' function."
        )

    def predict_batch(self, batch_x):
        """Prototype: predict_batch(self, batch_x, batch_y)

        Predicts on a single batch. Requires an outer control loop, feeding the data.
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'predict_batch()' function."
        )

    def save(self, folder):
        """Prototype: save(self, folder)

        Stores all model data inside the specified folder
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'save()' function."
        )

    def load(self, folder):
        """Prototype: save(self, folder)

        Loads all model data from the specified folder
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'load()' function."
        )
