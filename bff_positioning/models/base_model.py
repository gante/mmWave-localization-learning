""" Python module containing a class for basic interface of a ML model, as well as functions
shared by multiple model types.
"""

class ModelInterface():
    """ This class defines a common model interface. The models defined within this module
    should implement all depicted functions, to abstract the details away from the main scripts

    :param model_settings: a dictionary containing the model settings
    """

    # To add in the future:
    # 1) Instead of "train_batch()" with an outer control loop, create a "train_generator()" that
    #   uses generators
    # 2) Similarly, crate a "predict_generator()"
    # 3) Add an MC dropout flag to the predict functions, to enable uncertainty prediction

    def __init__(self, model_settings):
        self.model_settings = model_settings

    def check_settings(self, accepted_settings):
        """Checks the all input settings are usable. If they are not, it is likely that
        there was some misplanning with the model configuration, and it should be re-checked

        :param accepted_settings: list of strings with the accepted settings for each model
        """
        error_str = "Unexpected settings were found. Please double check the settings file!"\
            "\nList of expected settings: {}\nList of obtained settings: {}"
        assert set(self.model_settings.keys()) == set(accepted_settings), error_str.format(
            accepted_settings, list(self.model_settings.keys()))

    def setup(self):
        """Prototype: setup(self)

        Given the settings, sets up the model graph for training
        """
        raise NotImplementedError(
            "The model sub-class did not implement a 'setup()' function."
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
