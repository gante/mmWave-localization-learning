""" Python module with metric-related functions
"""

import sklearn.metrics as metrics
import numpy as np


SCORE_TYPES = ['accuracy', 'f1_score', 'mean_square_error', 'euclidean_distance']


def score_predictions(y_true, y_pred, score_type):
    """ Scores the model predictions against the ground truth, given the score type

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
