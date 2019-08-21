""" Python module containing a class for basic interface of an ML model, as well as functions
shared by multiple model types.
"""

import os
import logging
import numpy as np
import tensorflow as tf

from .layer_functions import add_linear_layer


class BaseModel():
    """ This class defines a common model interface. The models defined within this project
    should implement all depicted interface functions, to abstract the details away from the main
    scripts

    :param model_settings: a dictionary containing the model settings. All dictionary key/value
        pairs will be set as class members, for further readibility (and IDE auto-complete :D)
    """

    # To add in the future:
    # 1) Instead of "train_batch()" with an outer control loop, create a "train_generator()" that
    #   uses generators
    # 2) Similarly, crate a "predict_generator()"
    # 3) Add an MC dropout flag to the predict functions, to enable uncertainty prediction

    def __init__(self, model_settings):

        # Instatiates basic model settings
        self.input_name = model_settings["input_name"]
        self.input_shape = model_settings["input_shape"]
        self.output_name = model_settings["output_name"]
        self.output_type = model_settings["output_type"]
        self.output_shape = model_settings["output_shape"]
        self.batch_size = model_settings["batch_size"]
        self.batch_size_inference = model_settings["batch_size_inference"]
        self.dropout = model_settings["dropout"]
        self.epochs = model_settings["epochs"]
        self.fc_layers = model_settings["fc_layers"]
        self.fc_neurons = model_settings["fc_neurons"]
        self.learning_rate = model_settings["learning_rate"]
        self.learning_rate_decay = model_settings["learning_rate_decay"]

        # Instantiates other common variables that will be set later
        self.saver = None
        self.session = None
        self.train_step = None
        self.learning_rate_var = None
        self.keep_prob = None
        self.model_input = None
        self.model_target = None

        # Runs basic initializations
        self._set_gpu(model_settings)

    # ---------------------------------------------------------------------------------------------
    # Model interface functions
    def set_graph(self):
        """Prototype: setup(self)

        Given the settings, sets up the model graph for training
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'set_graph()' function."
        )

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

    def close(self):
        """Prototype: close(self)

        Cleans up the session and any left over data
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'close()' function."
        )

    # ---------------------------------------------------------------------------------------------
    # Non-interface functions: model input/output
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
            self.model_target = tf.placeholder(
                tf.float32,
                shape=[None] + self.output_shape,
                name=self.output_name
            )
        elif self.output_type == "classification":
            self.model_target = tf.placeholder(tf.int64, shape=[None], name=self.output_name)
        else:
            raise ValueError("Unknown 'output_type' ({}). Only 'classification' and 'regression' "
                "are accepted.".format(self.output_type))

    def _add_classification_output(self, input_data):
        """ Adds the last layer for classification models. Returns the trainable step.

        :param input_data: TF tensor with this layer's input
        :returns: the trainable step
        """
        # Defines the logits and the softmax output
        logits = add_linear_layer(input_data, self.output_shape)
        tf.nn.softmax(logits, name=self.output_name)

        # Defines the loss function [mean(cross_entropy(target_value - softmax))]
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.model_target,
            logits=logits
        ))

        # Defines the optimizer (ADAM) and the train step
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var)
        train_step = optimizer.minimize(cross_entropy)
        return train_step

    def _add_regression_output(self, input_data):
        """ Adds the last layer for regression models. Returns the trainable step.

        :param input_data: TF tensor with this layer's input
        :returns: the trainable step
        """
        # Defines the regression output (used to learn), and its clipped version (used to predict)
        regression = add_linear_layer(input_data, self.output_shape, bias=0.5)
        tf.clip_by_value(regression, 0.0, 1.0, name=self.output_name)

        # Defines the loss function [MSE = mean(square(target_value - regression))]
        mse = tf.reduce_mean(tf.square(self.model_target - regression))

        # Defines the optimizer (ADAM) and the train step
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var)
        train_step = optimizer.minimize(mse)
        return train_step

    # ---------------------------------------------------------------------------------------------
    # Non-interface functions: misc
    @staticmethod
    def _set_gpu(model_settings):
        """ If the option 'target_gpu' is defined in `model_settings`, sets that GPU
        """
        if "target_gpu" in model_settings:
            target_gpu = model_settings["target_gpu"]
            logging.info("[Using GPU #%s]", target_gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)

    def _prepare_model_for_training(self):
        """ Self-documenting :D
        """
        # Sets the saver
        self.saver = tf.train.Saver()

        # Sets the session
        self.session = tf.Session(config=tf.ConfigProto(gpu_options={"allow_growth": True}))

        # Initializes TF variables
        self.session.run(tf.global_variables_initializer())
        trainable_parameters = int(np.sum([np.product([var_dim.value for var_dim in var.get_shape()])
            for var in tf.trainable_variables()]))
        logging.info("Model initialized with %s trainable parameters!", trainable_parameters)
