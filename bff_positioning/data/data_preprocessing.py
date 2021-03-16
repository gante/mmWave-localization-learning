"""
Contains a class that converts the pre-processed binary file into a numpy array. The logic behind
each conversion step is thoroughly document through comments in the code.
"""

import pickle
import os
import struct
import logging
import hashlib
import numpy as np
from tqdm import tqdm
import matplotlib

class Preprocessor():
    """ Reads a pre-processed binary file (see README) into a pretty numpy array, which can be
    feed to a ML model.

    :param settings: a dictionary of data-related simulation settings
    """
    def __init__(self, settings):
        self.input_file = settings['input_file']
        self.preprocessed_file = settings['preprocessed_file']
        self.run_sanity_checks = settings['run_sanity_checks']

        # Inputs that dictate the dataset ID:
        self.max_time = settings['max_time']
        self.sample_freq = settings['sample_freq']
        self.beamformings = settings['beamformings']
        self.power_offset = settings['power_offset']
        self.power_scale = settings['power_scale']
        self.pos_grid = settings['pos_grid']
        self.pos_shift = settings['pos_shift']
        self.keep_timeslots = settings['keep_timeslots']

        # Precomputes other needed variables, and initializes others
        self.dataset_id = get_dataset_id(settings)
        self.time_slots = self.max_time * self.sample_freq
        self.features_size = self.time_slots * self.beamformings
        self.features = None
        self.labels = None

    def check_existing_dataset(self):
        """ Checks whether the dataset we are trying to create already exists

        :returns: Boolean flag, with `True` meaning that the dataset already exists
        """
        dataset_exists = False
        if os.path.isfile(self.preprocessed_file):
            with open(self.preprocessed_file, 'rb') as dataset_file:
                _, _, target_dataset_id = pickle.load(dataset_file)
            if target_dataset_id == self.dataset_id:
                dataset_exists = True
        return dataset_exists

    def create_bff_dataset(self):
        """ Creates a BFF experiments-ready dataset. The dataset contains `X`, the matrix
        containing the received radiation, and `y`, the true position for that received
        radiation.

        :returns: `X` and `y`. `X` is a matrix with dimentions (number_of_positions x
        radiation_samples_per_position). y is matrix with dimentions (number_of_positions x 2)
        The radiation samples per position's size is given by the number of used beamformings
        times the sampling frequency time the receiving time per beamforming
        """
        labels_size = 2
        sample_size = int(self.features_size + labels_size)

        # Converts the dataset into features/labels
        logging.info("Converting the dataset into features/labels...")
        features, labels = self._data_to_dataset(sample_size)

        # Converting the features/labels into numpy arrays
        logging.info("Converting features/labels into numpy arrays...")
        self.features = np.array(features)
        self.labels = np.array(labels)
        del features, labels

        # Printing the ranges of the features
        logging.info("[label] x range: %s - %s",
            self.labels[:, 0].min(), self.labels[:, 0].max())
        logging.info("[label] y range: %s - %s",
            self.labels[:, 1].min(), self.labels[:, 1].max())
        logging.info("[features] power range: %s - %s",
            self.features[:].min(), self.features[:].max())

        # Removes unwanted timeslots
        self._delete_timeslots()

        # Removes dataless positions
        self._remove_dataless_positions()

    def _data_to_dataset(self, sample_size):
        """ `create_bff_dataset` auxiliary function. Converts the raw (floating point)
        input data into features and labels, that will be further filtered. The features
        that come out of this operation should have a range close to [0, 1] -- please
        set the simulation parameters accordingly.

        :param sample_size: length of the input data for each position in the dataset
        """

        # Unpacks stored variables
        x_shift, y_shift = self.pos_shift
        x_grid, y_grid = self.pos_grid
        if x_grid != y_grid:
            logging.warning("WARNING: the area on which the experiments are going to be "
                "performes is not square. The distance metric during the model training "
                "will be ill defined. (e.g. the validation error in meters will not be correct)"
                "\nPlease keep that in mind (or add that exception to the code :D).")

        # Loads the binary mode dataset (the step that creates this binary data will
        # be rewritten in python in the near future)
        logging.info("Loading the binary dataset from %s...", self.input_file)
        with open(self.input_file, mode='rb') as file:
            data_binary = file.read()

        # Converts the binary dataset into float 32
        logging.info("Converting the binary data to float_32...")
        logging.info("[** this may take a couple of minutes and it will not print any progress **]")
        binary_size = os.path.getsize(self.input_file)
        size_bytes = int(binary_size/4)
        data = struct.unpack('f'*size_bytes, data_binary)
        del data_binary

        num_samples = int(size_bytes / sample_size)
        features = []
        labels = []

        # For each sample in the data
        logging.info("Creating features and labels from the float_32 data...")
        for sample_idx in tqdm(range(num_samples)):

            tmp_features = []
            tmp_labels = []
            data_start_pos = sample_idx * sample_size

            # For each data item in the sample
            # (0 = Position data - X)
            # (1 = Position data - Y)
            # (2, ..., sample_size-1 = Feature data)
            for data_idx in range(sample_size):
                item = data[data_start_pos + data_idx]
                if data_idx == 0:
                    item += x_shift
                    item /= x_grid
                    tmp_labels.append(item)
                elif data_idx == 1:
                    item += y_shift
                    item /= y_grid
                    tmp_labels.append(item)
                else:
            # Important notes regarding feature data:
            # 1) item == 0 -> there is no data here (there are no values > 0)
            # 2) The check for the minimum power threshold (e.g. -100 dBm) is performed
            #   after the noise is added, not here.
            # 3) Nevertheless, to speed up downstream operations code, filters out values with
            #   very little power. For the default simulation parameters, this filters samples
            #   with less than -170 dBm. Since the default "minimum_power" is -125 dBm [check
            #   an example for the meaning of this variable], this means we can reliably test
            #   (log-normal) noises with STD up to 15 dB [margin = (-125) - -170 = 45 dB =
            #   3*STD of 15 dB]
                    if -(self.power_offset) < item < 0:
                        tmp_features.append((item + self.power_offset) * self.power_scale)
                    else:
                        assert item <= 0.0, "There cannot be any value here above 0.0 (got {})"\
                            .format(item)
                        tmp_features.append(0.0)

            features.append(tmp_features)
            labels.append(tmp_labels)

        return features, labels

    def _delete_timeslots(self):
        """ Removes unwanted timeslots (Keep in mind that this feature's usefulness is super
        dataset-dependent! In my experiments, I removed the timeslots with very little data,
        corresponding to less than 1% of the non-zero features)
        """
        if self.keep_timeslots:
            logging.warning("Removing unwanted timeslots (keeping timeslots with indexes between"
                " '%s' and '%s')", self.keep_timeslots[0], self.keep_timeslots[1]-1)

            mask = np.ones(self.features.shape[1], dtype=bool)
            ts_to_keep = [ts for ts in range(*self.keep_timeslots)]
            ts_to_delete = [ts for ts in range(self.time_slots) if ts not in ts_to_keep]

            logging.info("Time slots to remove: %s", ts_to_delete)

            for idx in tqdm(range(self.features.shape[1])):
                # DIM 1 = BF, DIM 2 = TS
                if idx % self.time_slots in ts_to_delete:
                    mask[idx] = False

            # Removes those slots from the data
            logging.info("Shape before TS reduction: %s", self.features.shape)
            self.features = self.features[:, mask]
            logging.info("Shape after TS reduction: %s", self.features.shape)

    def _remove_dataless_positions(self):
        """ Removes invalid [x, y] positions (invalid positions = positions with no data,
        i.e. only zeroes)
        """
        logging.info("Detecting the invalid (data-less) positions... ")
        mask = np.ones(self.features.shape[0], dtype=bool)
        removed_pos = 0

        for idx in tqdm(range(self.features.shape[0])):
            if sum(self.features[idx, :]) == 0:
                mask[idx] = False
                removed_pos += 1

        self.features = self.features[mask, :]
        self.labels = self.labels[mask, :]
        logging.info("%s data-less positions removed.", removed_pos)

    def store_dataset(self):
        """ Stores the result of data preprocessing
        """
        # Final data reports
        logging.info("Usable positions: %s", self.features.shape[0])
        target_folder = os.path.split(self.preprocessed_file)[0]
        if not os.path.exists(target_folder):
            logging.info("Target folder (%s) not found, creating it...", target_folder)
            os.makedirs(target_folder)
        logging.info("Storing the result ...")
        with open(self.preprocessed_file, 'wb') as data_file:
            pickle.dump([self.features, self.labels, self.dataset_id], data_file)

        # Optional: plots the existing data points on a 2D image
        if self.run_sanity_checks:
            logging.info("Preparing plot to double-check existing data points...")
            # Creates (N+1) by (M+1) matrix. This means that its indexes go from 0 through N/M
            to_plot = np.full([int(self.pos_grid[0]) + 1, int(self.pos_grid[1]) + 1], 0.0)
            for pos_idx in tqdm(range(self.labels.shape[0])):
                # Scales 0-1 to 0-N/M
                pos_x = int(round(self.labels[pos_idx, 0] * self.pos_grid[0]))
                pos_y = int(round(self.labels[pos_idx, 1] * self.pos_grid[1]))
                # Flips Y (to correctly plot with imshow)
                to_plot[pos_x, int(round(self.pos_grid[1] - pos_y))] = 1.0
            # Local import to avoid messing non-gaphical interfaces
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            # Saves image and corresponding data
            valid_positions_img = np.transpose(to_plot)
            np.save(os.path.join(target_folder, 'existing_data_points.npy'), valid_positions_img)
            plt.imshow(valid_positions_img)
            image_locaton = os.path.join(target_folder, 'existing_data_points.pdf')
            plt.savefig(image_locaton)
            logging.info("Done! (Check %s)", image_locaton)

    def load_dataset(self, power_in_dbm=False):
        """ Loads the previously stored dataset, returning it

        :param power_in_dbm: reverts the transformed features to dbm units
        :returns: previously stored features and labels
        """
        assert self.check_existing_dataset(), "The dataset with the specified path ({}) either "\
            "does not exists or was built with different simulation settings. Please run the "\
            "data preprocessing step with the new simulation settings!".format(
            self.preprocessed_file)

        with open(self.preprocessed_file, 'rb') as dataset_file:
            features, labels, _ = pickle.load(dataset_file)

        if power_in_dbm:
            features = np.asarray(features)
            features /= self.power_scale
            features = np.where(features == 0.0, 0.0, features - self.power_offset)
            features = features.tolist()

        return features, labels


def get_dataset_id(settings):
    """ Creates and returns an unique ID (for practical purposes), given the data parameters.
    The main use of this ID is to make sure we are using the correct data source,
    and that the data parameters weren't changed halfway through the simulation sequence.
    """
    hashing_features = [
        settings['max_time'],
        settings['sample_freq'],
        settings['beamformings'],
        settings['power_offset'],
        settings['power_scale'],
        settings['pos_grid'],
        settings['pos_shift'],
        settings['keep_timeslots'],
    ]

    hash_sha256 = hashlib.sha256()
    for feature in hashing_features:
        if isinstance(feature, list):
            inner_features = feature
        else:
            inner_features = [feature]

        for item in inner_features:
            if isinstance(item, float):
                item = "{:.4f}".format(item)
                hash_sha256.update(bytes(item, encoding='utf8'))
            else:
                hash_sha256.update(bytes(item))

    return hash_sha256.hexdigest()
