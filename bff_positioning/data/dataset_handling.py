""" Python module that contains functions to manipulate the datasets used to train the models
"""

import logging
import concurrent.futures
import numpy as np

from sklearn.preprocessing import Binarizer, Normalizer


def create_noisy_features(
    features,
    labels,
    experiment_settings,
    data_parameters,
):
    """
    Creates ONE noisy sample (i.e. pair of noisy features [received power] with non-noisy labels
    [position]) PER POSITION, given the target noise level. Any noisy feature instance below the
    detection threshold is discarded (i.e. set to 0).
    Note: the features here should be in range [0, ~1.2], according to the original experiments

    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param labels: numpy 2D matrix, [sample_index, dimention]
    :param experiment_settings: experiment-related settings
    :param data_parameters: raw data-related settings
    :return: one full set of noisy features, with the corresponding non-noisy labels
    """
    # Computes some auxiliary variables
    scaler = _get_scaler(experiment_settings["scaler_type"])
    scaled_noise, scaled_cutoff = _convert_power_variables(
        experiment_settings,
        data_parameters
    )

    # Shortcut: no noise to be added? return original data
    if scaled_noise == 0.0:
        return features, labels

    # Adds noise
    noise = np.random.normal(scale=scaled_noise, size=features.shape)
    noisy_features = features + noise

    # Cuts features below the minimum power detection threshold
    noisy_features[noisy_features < scaled_cutoff] = 0

    # Removes the samples containing only 0s as features
    mask = np.ones(labels.shape[0], dtype=bool)
    for idx in range(labels.shape[0]):
        this_samples_sum = np.sum(noisy_features[idx, :])
        if this_samples_sum < 0.01:
            mask[idx] = False
    noisy_features = noisy_features[mask, :]
    noisy_labels = labels[mask, :]

    # Sanity check
    assert noisy_features.shape[0] == noisy_labels.shape[0]
    assert noisy_labels.shape[1] == 2
    assert noisy_features.shape[1] == features.shape[1]

    # Applies the scaler, if wanted
    if scaler is not None:
        noisy_features = scaler.fit_transform(noisy_features)

    # If the model is a cnn, reshapes the input
    if experiment_settings["model_type"] == "cnn":
        beamformings = data_parameters["beamformings"]
        time_slots = int(features.shape[1] / beamformings)
        noisy_features = np.reshape(
            noisy_features, (noisy_features.shape[0], beamformings, time_slots, 1)
        )

    return noisy_features, noisy_labels


def _convert_power_variables(experiment_settings, data_parameters):
    """ `create_noisy_features` auxiliary function. Scales some power-related settings by
    as much as the features were scaled.

    :param experiment_settings: [description]
    :param data_parameters: [description]
    :return: noise and power cut off variables, as used in `create_noisy_features`
    """
    # Unpacks variables
    power_offset = data_parameters["power_offset"]
    power_scale = data_parameters["power_scale"]
    original_tx_power = data_parameters["original_tx_power"]
    original_rx_gain = data_parameters["original_rx_gain"]
    baseline_cut = experiment_settings["detection_threshold"]
    tx_power = experiment_settings["tx_power"]
    rx_gain = experiment_settings["rx_gain"]

    # Computes and scales the detection theshold
    adjusted_cutoff = baseline_cut - (tx_power - original_tx_power) - (rx_gain - original_rx_gain)
    scaled_cutoff = (adjusted_cutoff + power_offset) * power_scale

    # Scales the noise
    scaled_noise = experiment_settings["noise_std"] * power_scale

    return scaled_noise, scaled_cutoff


def _get_scaler(scaler_type):
    """Returns a scaler to apply to the features.

    :param binary_scaler: toggle to select between Binarizer and Normalizer scaler,
        defaults to True (Binarizer)
    :return: the initialized scaler, and its name
    """
    scaler = None
    if scaler_type == "binarizer":
        scaler = Binarizer(threshold=0.1, copy=False)
    elif scaler_type == "normalizer":
        scaler = Normalizer(copy=False)
    elif scaler_type is not None:
        raise ValueError("Invalid scaler type ({})! Accepted values: 'binarizer', 'normalizer'"\
            .format(scaler_type))
    return scaler


def undersample_bf(features, beamformings):
    """ Halves the number of beamformings used in the features (expected use: 32 -> 16 BF)

    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param beamformings: Number of beamformings used to create the BFF
    :returns: Updated features
    """
    time_slots = int(features.shape[1] / beamformings)
    mask = np.ones(time_slots * beamformings, dtype=bool)
    for idx in range(time_slots * beamformings):
        #DIM 1 = BF, DIM 2 = TS
        if (idx//time_slots)%2 == 0:
            mask[idx] = False
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
    assert distance >= 1, "The minimum distance between samples has to be an integer "\
        "equal or greater than 1"

    mask = np.ones(labels.shape[0], dtype=bool)
    for idx in range(labels.shape[0]):
        label_x_scaled = int(labels[idx, 0] * 400)
        if label_x_scaled % distance > 0:
            mask[idx] = False
        else:
            label_y_scaled = int(labels[idx, 1] * 400)
            if label_y_scaled % distance > 0:
                mask[idx] = False

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

    for idx in range(labels.shape[0]):

        x_index = int(np.floor(labels[idx, 0] * lateral_partition))
        if x_index == lateral_partition:
            x_index = lateral_partition - 1

        y_index = int(np.floor(labels[idx, 1] * lateral_partition))
        if y_index == lateral_partition:
            y_index = lateral_partition - 1

        true_index = (y_index * lateral_partition) + x_index
        class_indexes.append(true_index)

    class_indexes = np.asarray(class_indexes)

    return class_indexes


def get_95th_percentile(y_true, y_pred, rescale_factor=1.):
    """ Gets the 95th percentile for the distance

    :param y_true: ground truth
    :param y_pred: model predictions
    """
    len_pred = y_pred.shape[0]
    array_of_distances = np.sqrt(np.sum(np.square(y_true[:len_pred, :] - y_pred), 1))
    return np.percentile(array_of_distances, 95) * rescale_factor

# -------------------------------------------------------------------------------------------------
# Path-handling functions

def _static_paths_sampler(
    mask,
    paths,
    features,
    labels,
    time_steps,
    experiment_settings,
    data_parameters
):
    """ Helper function to `sample_paths` - samples static paths
    Note - static paths format = {(x, y): index in the dataset}
    """
    X, y = [], []
    list_of_wanted_positions = np.asarray(list(paths.values()))[mask]

    def _process_a_path(index_in_dataset):
        x_sequence, label = None, None
        x_sequence = [features[index_in_dataset, :]]*time_steps
        x_sequence = _apply_noise_and_scaler(
            np.asarray(x_sequence),
            experiment_settings,
            data_parameters
        )
        if x_sequence is not None:
            label = labels[index_in_dataset, :]
        return x_sequence, label

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x_sequence, label in executor.map(_process_a_path, list_of_wanted_positions):
            if label is not None:
                X.append(x_sequence)
                y.append(label)

    return np.asarray(X), np.asarray(y)


def _moving_paths_sampler(
    mask,
    paths,
    features,
    labels,
    time_steps,
    experiment_settings,
    data_parameters
):
    """ Helper function to `sample_paths` - samples moving paths
    Note - moving paths format = [[dataset index for pos_1, dataset index for pos_2, ...], [...]]
    """
    X, y = [], []
    list_of_wanted_sequences = np.asarray(paths)[mask, :]

    def _process_a_path(sequence_of_indexes):
        x_sequence, label = [], None
        for dataset_index in sequence_of_indexes:
            x_sequence.append(features[dataset_index, :])
        assert len(x_sequence) == time_steps, "The length of the obtained sequence ({}) does "\
            "not match the expected length ({})".format(len(x_sequence), time_steps)
        x_sequence = _apply_noise_and_scaler(
            np.asarray(x_sequence),
            experiment_settings,
            data_parameters
        )
        if x_sequence is not None:
            label = labels[sequence_of_indexes[-1], :]
        return x_sequence, label

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x_sequence, label in executor.map(_process_a_path, list_of_wanted_sequences):
            if label is not None:
                X.append(x_sequence)
                y.append(label)

    return np.asarray(X), np.asarray(y)


def _apply_noise_and_scaler(
    sequence,
    experiment_settings,
    data_parameters
):
    """ Helper function to `sample_paths` - applies the noise and the scaler over a sequence
    of features
    """

    scaled_noise, scaled_cutoff = _convert_power_variables(
        experiment_settings,
        data_parameters
    )
    noise = np.random.normal(scale=scaled_noise, size=sequence.shape)
    noisy_sequence = sequence + noise
    noisy_sequence[noisy_sequence < scaled_cutoff] = 0

    # Checks if the sequence contains any empty sample (sample containing only 0).
    # If it does, discards the sequence, as it is a broken sequence.
    this_sample_sums = np.sum(noisy_sequence, axis=1)
    this_sample_sums[this_sample_sums > 0.001] = 1
    if np.sum(this_sample_sums) < len(sequence):
        return None

    #Applies the scaler
    scaler = _get_scaler(experiment_settings["scaler_type"])
    if scaler is not None:
        noisy_sequence = scaler.fit_transform(noisy_sequence)
        if experiment_settings["scaler_type"] == "binarizer":
            noisy_sequence = noisy_sequence.astype(bool)
    return noisy_sequence


def sample_paths(
    paths,
    features,
    labels,
    experiment_settings,
    data_parameters,
    path_parameters,
    sample_fraction=1.
):
    """
    Given the input arguments (see description below), returns:
        1 - X, the noisy sequence input data (sequences with the predefined length)
        2 - y, the labels (one label per sequence)
        3 - a dict with the ending index for that path type (used at test time)

    Having RAM problems? Use the "sample_fraction" option in `path_parameters` and train for
    more "epochs" (in this case, epochs is not the most adequate word :D) The function will use
    ~sample_fraction times the original dataset per "epoch", were 0 <= sample_fraction < 1

    :param paths: paths created in the preprocessing step. The paths will be used with `features`
        to create the actual dataset
    :param features: numpy 2D matrix, [sample_index, feature_index]
    :param labels: numpy 2D matrix, [sample_index, dimention]
    :param experiment_settings: experiment-related settings
    :param data_parameters: raw data-related settings
    :param path_parameters: path-related settings
    :param sample_fraction: floating point between 0 and 1, indicating the fraction of paths to be
        sampled
    :returns: see the list above
    """
    # Unpacks a few arguments
    time_steps = path_parameters["time_steps"]

    X = None
    y = None
    paths_type_delimiter = []
    assert 0. <= sample_fraction <= 1., "sample_fraction should be between 0.0 and 1.0!"
    assert labels.shape[0] == features.shape[0], "The features and the labels must have the same "\
        "input length!"

    def _process_path_type(path_tuple):
        """ path_tuple: (path_type, paths_for_this_type) """
        # Creates a probabilistic mask (skip paths with probability 1-sample_fraction)
        mask = np.random.uniform(size=len(path_tuple[1]))
        mask[mask < sample_fraction] = 1.0
        mask[mask < 1.0] = 0.0

        # Samples a sequences with size=time_steps.
        if path_tuple[0] == 's':
            x_path, y_path = _static_paths_sampler(
                mask.astype(bool),
                path_tuple[1],
                features,
                labels,
                time_steps,
                experiment_settings,
                data_parameters
            )
        else:
            assert path_tuple[0] in ('p', 'c')
            x_path, y_path = _moving_paths_sampler(
                mask.astype(bool),
                path_tuple[1],
                features,
                labels,
                time_steps,
                experiment_settings,
                data_parameters
            )
        return x_path, y_path, path_tuple[0]

    for x_path, y_path, path_type in map(_process_path_type, paths.items()):
        paths_type_delimiter.append((path_type, y_path.shape[0]))
        if X is None:
            X = x_path
            y = y_path
        else:
            X = np.append(X, x_path, axis=0)
            y = np.append(y, y_path, axis=0)

    return np.asarray(X), np.asarray(y), paths_type_delimiter
