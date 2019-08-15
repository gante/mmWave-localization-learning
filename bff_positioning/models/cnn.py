""" Python class depicting a Convolutional Neural Network (CNN).
"""

from .base_model import ModelInterface

ACCEPTED_SETTINGS = [
    "hidden_layers",
    "cnn_layers",
]

class CNN(ModelInterface):
    """Convolutional Neural Network class

    :param model_settings: a dictionary containing the model settings
    """

    def __init__(self, model_settings):
        super().__init__(model_settings=model_settings)
        self.check_settings(ACCEPTED_SETTINGS)
