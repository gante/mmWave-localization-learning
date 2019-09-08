""" Python module containing a class for basic interface of an ML model, as well as functions
shared by multiple model types.
"""

import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import sklearn.metrics as metrics

import tensorflow as tf
from keras_radam.training import RAdamOptimizer

from .layer_functions import add_linear_layer

SCORE_TYPES = ['accuracy', 'f1_score', 'mean_square_error', 'euclidean_distance']


# -------------------------------------------------------------------------------------------------
# Workaround TF logger problem in TF 1.14
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)  #pylint: disable=protected-access
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False               #pylint: disable=protected-access
except Exception:                                           #pylint: disable=broad-except
    pass
# Workaround TF logger problem in TF 1.14
# -------------------------------------------------------------------------------------------------


class BaseModel():
    """ This class defines a common model interface. The models defined within this project
    should explicitly implement all depicted interface functions, to abstract the details
    away from the main scripts.

    :param model_settings: a dictionary containing the model settings. All dictionary key/value
        pairs will be set as class members, for further readibility (and IDE auto-complete :D)
    """

    # To add in the future:
    # - Add TensorBoard functionality
    # - Instead of "train_batch()" with an outer control loop, create a "train_generator()" that
    #   uses generators
    # - Similarly, crate a "predict_generator()"
    # - Add an MC dropout flag to the predict functions, to enable uncertainty prediction

    def __init__(self, model_settings):

        # Instatiates basic model settings
        self.model_folder = model_settings["model_folder"]
        self.input_name = model_settings["input_name"]
        self.input_shape = model_settings["input_shape"]
        self.output_name = model_settings["output_name"]
        self.output_type = model_settings["output_type"]
        self.output_shape = model_settings["output_shape"]
        self.validation_metric = model_settings["validation_metric"]
        self.batch_size = model_settings["batch_size"]
        self.batch_size_inference = model_settings["batch_size_inference"]
        self.dropout = model_settings["dropout"]
        self.early_stopping = model_settings["early_stopping"]
        self.max_epochs = model_settings["max_epochs"]
        self.fc_layers = model_settings["fc_layers"]
        self.fc_neurons = model_settings["fc_neurons"]
        self.learning_rate = model_settings["learning_rate"]
        self.learning_rate_decay = model_settings["learning_rate_decay"]
        self.optimizer_type = model_settings["optimizer_type"]

        # Instantiates other common variables that might be used later
        self.current_validation_score = None
        self.current_learning_rate = None
        self.current_epoch = None
        self.epochs_not_improving = None
        self.best_validation_score = None
        self.saver = None
        self.session = None
        self.train_step = None
        self.learning_rate_var = None
        self.dropout_var = None
        self.model_input = None
        self.model_output = None
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

    def train_epoch(self, X, Y):
        """Prototype: train_batch(self, X, Y)

        Trains on a single epoch, given the data (as numpy variables).
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'train_epoch()' function."
        )

    def epoch_end(self, X=None, Y=None):
        """Prototype: epoch_end(self, X, Y)

        Executes end of epoch logic (e.g. decay learning rate, try early stopping, ...) and
        returns whether the model should keep training or not
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'epoch_end()' function."
        )

    def predict(self, X):
        """Prototype: predict(self, X)

        Returns the preditions on the given data
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'predict()' function."
        )

    def save(self, model_name):
        """Prototype: save(self, model_name)

        Stores all model data inside the specified folder, given the model name
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'save()' function."
        )

    def load(self, model_name):
        """Prototype: load(self, model_name)

        Loads all model data from the specified folder, given the model name
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
    # Non-interface functions: model training
    def _train_epoch(self, X, Y):
        """Default training routine - trains the model for an epoch

        :param X: numpy array with the features
        :param Y: numpy array with the labels
        """
        assert X.shape[0] == Y.shape[0], "X and Y have a different number of samples!"
        max_batches = int(np.ceil(X.shape[0] / self.batch_size))
        X, Y = shuffle(X, Y)
        train_string = "Training on epoch: {:4} || LR: {:3.2E} ||"

        for batch_idx in tqdm(
            range(max_batches),
            desc=train_string.format(self.current_epoch, self.current_learning_rate)
        ):
            start_batch = batch_idx * self.batch_size
            end_batch = min((batch_idx + 1) * self.batch_size, X.shape[0])
            self.train_step.run(
                feed_dict={
                    self.model_input: X[start_batch:end_batch, ...],
                    self.model_target: Y[start_batch:end_batch, ...],
                    self.dropout_var: self.dropout,
                    self.learning_rate_var: self.current_learning_rate,
                },
                session=self.session)
        self.current_epoch += 1

    def _epoch_end(self, y_true=None, y_pred=None):
        """ Default end of epoch routine - Performs end of epoch operations, such as decaying the
        learning rate. Some operations, such as the early stopping, require a validation set.

        :param y_true: ground truth, defaults to None
        :param y_pred: model predictions, defaults to None
        :returns: boolean indicating whether the model should keep training, validation score
        """
        keep_training = True
        val_score = None
        if self.current_epoch >= self.max_epochs:
            keep_training = False
        else:
            # Decays LR
            self.current_learning_rate *= self.learning_rate_decay
            if y_true is not None and y_pred is not None:
                val_score = score_predictions(y_true, y_pred, self.validation_metric)
                if self.early_stopping:
                    # Evaluates early stopping, if requested
                    keep_training = self._eval_early_stopping(val_score)
        return keep_training, val_score

    def _predict(self, X):
        """ Returns the predictions on the given data

        :param X: numpy array with the features
        :return: an numpy array with the predictions
        """
        max_batches = int(np.ceil(X.shape[0] / self.batch_size_inference))
        predictions = []
        for batch_idx in range(max_batches):
            start_batch = batch_idx * self.batch_size_inference
            end_batch = min((batch_idx + 1) * self.batch_size_inference, X.shape[0])
            batch_predictions = self.model_output.eval(
                feed_dict={
                    self.model_input: X[start_batch:end_batch, ...],
                    self.dropout_var: 0.0,
                },
                session=self.session
            )
            batch_size = end_batch - start_batch
            predictions.extend([batch_predictions[idx, ...] for idx in range(batch_size)])
        return np.asarray(predictions)

    def _eval_early_stopping(self, validation_score):
        """ Evaluates the early stopping mechanism

        :param validation_score: the validation score
        :return: boolean depicting whether the model should keep training
        """
        keep_training = True
        if self.current_validation_score is None:
            self.current_validation_score = validation_score
            self.epochs_not_improving = 0
        elif validation_score < self.current_validation_score:
            self.current_validation_score = validation_score
            self.epochs_not_improving = 0
        else:
            self.epochs_not_improving += 1
        if self.epochs_not_improving >= self.early_stopping:
            logging.info("Early stopping - the model does not improve for %s epochs",
                self.epochs_not_improving)
            keep_training = False
        return keep_training

    # ---------------------------------------------------------------------------------------------
    # Non-interface functions: model input/output
    def _set_graph_io(self):
        """ Auxiliary function to "set_graph()". Sets the graph input and trainable target,
        as well as some other basic variables common to all model types
        """
        # The current learning rate
        self.learning_rate_var = tf.compat.v1.placeholder(tf.float32, shape=[])
        # Dropout probability
        self.dropout_var = tf.compat.v1.placeholder(tf.float32, name='dropout')

        self.model_input = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None] + self.input_shape,
            name=self.input_name,
        )
        if self.output_type == "regression":
            self.model_target = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None] + self.output_shape
            )
        elif self.output_type == "classification":
            self.model_target = tf.compat.v1.placeholder(
                tf.int64,
                shape=[None]
            )
        else:
            raise ValueError("Unknown 'output_type' ({}). Only 'classification' and 'regression' "
                "are accepted.".format(self.output_type))

    def _add_classification_output(self, input_data):
        """ Adds the last layer for classification models. Returns the trainable step.

        :param input_data: TF tensor with this layer's input
        :returns: the trainable step
        """
        # Defines the logits and the softmax output
        assert len(self.output_shape) == 1, "This function is not ready for outputs with higher "\
            "dimensionality"
        logits = add_linear_layer(input_data, self.output_shape[0])
        self.model_output = tf.nn.softmax(logits, name=self.output_name)

        # Defines the loss function [mean(cross_entropy(target_value - softmax))]
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.model_target,
            logits=logits
        ))

        # Defines the optimizer and the train step
        optimizer = self._set_optimizer()
        train_step = optimizer.minimize(cross_entropy)
        return train_step

    def _add_regression_output(self, input_data):
        """ Adds the last layer for regression models. Returns the trainable step.

        :param input_data: TF tensor with this layer's input
        :returns: the trainable step
        """
        # Defines the regression output (used to learn), and its clipped version (used to predict)
        assert len(self.output_shape) == 1, "This function is not ready for outputs with higher "\
            "dimensionality"
        regression = add_linear_layer(input_data, self.output_shape[0], bias=0.5)
        self.model_output = tf.clip_by_value(regression, 0.0, 1.0, name=self.output_name)

        # Defines the loss function [MSE = mean(square(target_value - regression))]
        mse = tf.reduce_mean(tf.square(self.model_target - regression))

        # Defines the optimizer and the train step
        optimizer = self._set_optimizer()
        train_step = optimizer.minimize(mse)
        return train_step

    def _set_optimizer(self):
        """ Returns a TF optimizer, given the model settings
        """
        if self.optimizer_type == "ADAM":
            return tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_var)
        elif self.optimizer_type == "NADAM":
            return tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate_var)
        elif self.optimizer_type == "RADAM":
            return RAdamOptimizer(learning_rate=self.learning_rate_var)
        else:
            raise ValueError("{} is not a supported optimizer type. Supported optimizer types: "
                "ADAM, NADAM, RADAM.".format(self.optimizer_type))

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
        # Sets the saver and the session
        self.saver = tf.compat.v1.train.Saver()
        self._set_session()

        # Initializes TF variables
        self.session.run(tf.compat.v1.global_variables_initializer())
        trainable_parameters = int(np.sum(
            [np.product([var_dim.value for var_dim in var.get_shape()])
            for var in tf.compat.v1.trainable_variables()]
        ))
        logging.info("Model initialized with %s trainable parameters!", trainable_parameters)

        # Initializes other train-related variables
        self.current_learning_rate = self.learning_rate
        self.current_epoch = 0

    def _save(self, model_name):
        """ Default function to save a model

        :param model_name: the name of the model
        """
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        save_path = os.path.join(self.model_folder, model_name)
        self.saver.save(self.session, save_path)

    def _load(self, model_name):
        """ Default function to load a model

        :param model_name: the name of the model
        """
        self._set_session()
        loader_path = os.path.join(self.model_folder, model_name)
        loader = tf.compat.v1.train.import_meta_graph(loader_path + '.meta')
        loader.restore(self.session, loader_path)

        # Redefines key model graph-related variables
        # (e.g. placeholders and operations that were saved by name)
        graph = tf.compat.v1.get_default_graph()
        self.model_input = graph.get_tensor_by_name(self.input_name + ":0")
        self.model_output = graph.get_tensor_by_name(self.output_name + ":0")
        self.dropout_var = graph.get_tensor_by_name("dropout:0")

    def _set_session(self):
        """ Default function to set the session
        """
        self.session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options={"allow_growth": True})
        )

    def _close_session(self):
        """ Default function to close the session
        """
        self.session.close()


def score_predictions(y_true, y_pred, score_type):
    """ Scores the model predictions agains the ground truth, given the score type

    :param y_true: ground truth
    :param y_pred: model predictions
    :param score_time: the type of score
    :retuns: the predictions' score
    """
    score = None
    if score_type not in SCORE_TYPES:
        raise ValueError("{} is not a valid score type. Implemented score types: {}".format(
            score_type, SCORE_TYPES))
    if score_type == 'accuracy':
        score = metrics.accuracy_score(y_true, y_pred)
    elif score_type == 'f1_score':
        score = metrics.f1_score(y_true, y_pred)
    elif score_type == 'mean_square_error':
        score = metrics.mean_squared_error(y_true, y_pred)
    elif score_type == 'euclidean_distance':
        score = np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), 1)))
    return score
