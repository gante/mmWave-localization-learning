""" Python class depicting a Deep Neural Network (DNN).
"""
# pylint: disable=no-member

from .base_model import ModelInterface, BASE_SETTINGS

ACCEPTED_SETTINGS = BASE_SETTINGS + [
]

class DNN(ModelInterface):
    """Deep Neural Network class

    :param model_settings: a dictionary containing the model settings
    """

    def __init__(self, model_settings):
        super().__init__(model_settings=model_settings)
        self._check_settings_names(ACCEPTED_SETTINGS)
        self._set_gpu()
