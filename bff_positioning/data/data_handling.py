""" Python module that contains functions to manipulate data
"""

import math
import logging
import numpy as np

from sklearn.preprocessing import Binarizer, Normalizer


def get_scaler(binary_scaler=True):
    """Returns a scaler to apply to the features.

    :param binary_scaler: toggle to select between Binarizer and Normalizer scaler,
        defaults to True (Binarizer)
    :return: the initialized scaler, and its name
    """
    if binary_scaler:
        scaler = Binarizer(0.1, copy=False)
        scaler_name = 'binarized'
    else:
        scaler = Normalizer(copy=False)
        scaler_name = 'normalized'
    return scaler, scaler_name


def create_noisy_features(
    features,
    labels,
    noise_std_converted,
    min_pow_cutoff,
    scaler=None,
):
    """
    Creates ONE noisy sample (i.e. pair of noisy features [received power] with non-noisy labels
    [position]) PER POSITION, given the target noise level. Any noisy feature instance below the
    detection threshold is discarded (i.e. set to 0).
    Note: the features here should be in range [0, ~1.2], according to the original experiments

    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param labels: numpy 2D matrix, [sample_index, dimention]
    :param noise_std_converted: log-normal noise to add, with the same down-scale as the features
    :param min_pow_cutoff: minimum accepted power for the features, that will be applied after the
        noise.
    :param scaler: scales the features according to the scaler (check `get_scaler`)
    :return: one full set of noisy features, with the corresponding non-noisy labels
    """
    # Adds noise
    noise = np.random.normal(scale=noise_std_converted, size=features.shape)
    noisy_features = features + noise

    # Cuts features below the minimum power detection threshold
    noisy_features[noisy_features < min_pow_cutoff] = 0

    # Removes the samples containing only 0s as features
    mask = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        this_samples_sum = np.sum(noisy_features[i, :])
        if this_samples_sum < 0.01:
            mask[i] = False
    noisy_features = noisy_features[mask, :]
    noisy_labels = labels[mask, :]

    # Sanity check
    assert noisy_features.shape[0] == noisy_labels.shape[0]
    assert noisy_labels.shape[1] == 2
    assert noisy_features.shape[1] == features.shape[1]

    # Applies the scaler, if wanted
    if scaler is not None:
        noisy_features = scaler.fit_transform(noisy_features)

    return noisy_features, noisy_labels


def undersample_bf(features, time_slots, beamformings):
    """ Halves the number of beamformings used in the features (expected use: 32 -> 16 BF)

    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param time_slots: Number of samples per beamforming index
    :param beamformings: Number of beamformings used to create the BFF
    :returns: Updated features
    """
    mask = np.ones(time_slots * beamformings, dtype=bool)
    for i in range(time_slots * beamformings):
        #DIM 1 = BF, DIM 2 = TS
        if (i//time_slots)%2 == 0:
            mask[i] = False
    features = features[:, mask]
    logging.warning("Attention: features undersampled to 16 BFs. Features shape: %s",
        features.shape)
    return features


def undersample_space(features, labels, distance):
    """ Widens the space between samples.

    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param labels: numpy 2D matrix, [sample_index, dimention]
    :param distance: minimum distance between samples (in meters, min=1m)
    :returns: Updated features and labels
    """
    distance = int(distance) #just in case

    mask = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        label_x_scaled = int(labels[i, 0] * 400)
        if label_x_scaled % distance > 0:
            mask[i] = False
        else:
            label_y_scaled = int(labels[i, 1] * 400)
            if label_y_scaled % distance > 0:
                mask[i] = False

    features = features[mask, :]
    labels = labels[mask, :]

    logging.warning("Attention: the minimum distance between samples is now %s meters. "
        "Features shape: %s", distance, features.shape)
    return features, labels


def position_to_class(labels, lateral_partition):
    """ Used with hierarchical CNN experiments.
    Converts a list of 2D positions into a list of classes, given a lateral partition. Of course,
    this assumes the 2D area is a square, and each resulting class will be a sub-square with
    side = lateral_partition. In other words, if lateral_partition = N, the original area will be
    split in N^2 classes

    :param labels: numpy 2D matrix, [sample_index, dimention]
    :param lateral_partition: number of lateral partitions
    """
    class_indexes = []

    for i in range(labels.shape[0]):

        x_index = int(math.floor(labels[i, 0] * lateral_partition))
        if x_index == lateral_partition:
            x_index = lateral_partition - 1

        y_index = int(math.floor(labels[i, 1] * lateral_partition))
        if y_index == lateral_partition:
            y_index = lateral_partition - 1

        true_index = (y_index * lateral_partition) + x_index
        class_indexes.append(true_index)

    class_indexes = np.asarray(class_indexes)

    return class_indexes
