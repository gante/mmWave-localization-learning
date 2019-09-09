"""
Contains a class that takes the dataset's possible positions and a few path-related options,
and creates train, validation, and test paths for the tracking experiments
"""

import os
import logging
import hashlib
import pickle
import numpy as np
from tqdm import tqdm

PI = np.pi


class PathCreator():
    """ Creates paths for tracking experiments, given the possible positions and a few options

    :param data_settings: dictionary of data-related simulation settings
    :param path_settings: dictionary of path-related settings
    :param labels: numpy 2D matrix, [sample_index, dimention]
    """
    def __init__(self, data_settings, path_settings, labels):
        self.paths_file = path_settings['paths_file']

        # Inputs that dictate the dataset ID:
        self.pos_grid = data_settings['pos_grid']
        self.time_steps = path_settings['time_steps']
        self.moving_paths_multiplier = path_settings['moving_paths_multiplier']

        self.s_paths = path_settings['s_paths']

        self.p_paths = path_settings['p_paths']
        self.p_avg_speed = path_settings['p_avg_speed']
        self.p_max_speed = path_settings['p_max_speed']
        self.p_acceleration = path_settings['p_acceleration']
        self.p_direction_change = path_settings['p_direction_change']
        self.p_move_proba = path_settings['p_move_proba']

        self.c_paths = path_settings['c_paths']
        self.c_avg_speed = path_settings['c_avg_speed']
        self.c_max_speed = path_settings['c_max_speed']
        self.c_acceleration = path_settings['c_acceleration']
        self.c_direction_change = path_settings['c_direction_change']
        self.c_move_proba = path_settings['c_move_proba']

        # Declares other variables
        self.paths_id = get_paths_id(data_settings, path_settings)
        self.valid_positions = self._get_valid_positions(labels)
        self._paths = None

    def _get_valid_positions(self, labels):
        """ Returns a dictionary with all valid possitions, and their index in the original
        dataset (a lookup table).

        :param positions: numpy 2D matrix, [sample_index, dimention]
        :returns: dictionary with shape {(x,y) : index}
        """
        valid_positions = {}
        for idx in range(labels.shape[0]):
            this_x = round(labels[idx, 0] * self.pos_grid[0])
            this_y = round(labels[idx, 1] * self.pos_grid[1])
            assert this_x >= 0
            assert this_y >= 0
            assert (this_x, this_y) not in valid_positions, "There can be no repeated entries"
            valid_positions[(this_x, this_y)] = idx
        logging.info("Number of valid positions: %s", len(valid_positions))
        x = [position[0] for position in valid_positions]
        y = [position[1] for position in valid_positions]
        logging.info("Range of valid positions after upscaling: x=%s-%s, y=%s-%s",
            min(x), max(x), min(y), max(y))
        return valid_positions

    def _get_static_paths(self, undersample):
        """ Creates static path data

        :param undersample: floating point between 0 and 1 that undersamples the dictionary of
            paths, based on this ratio
        """
        logging.info("Getting static paths...")
        if undersample < 1.:
            mask = np.where(
                np.random.uniform(size=len(self.valid_positions)) < undersample,
                1.0,
                0.0
            )
            paths = {
                key: value for idx, (key, value) in enumerate(self.valid_positions.items())\
                    if mask[idx]
            }
        else:
            paths = self.valid_positions
        return paths

    def _get_moving_paths(self, path_type, undersample):
        """ Creates moving path data

        :param path_type: str depicting the type of path ('p' for pedestrian, 'c' for car)
        :param undersample: floating point between 0 and 1 that undersamples the dictionary of
            paths, based on this ratio
        """
        # Sets movement options for the desired type (also converts degrees to radians)
        assert path_type in ('c', 'p'), "Invalid path type!"
        avg_speed = getattr(self, path_type + '_avg_speed')
        max_speed = getattr(self, path_type + '_max_speed')
        acceleration = getattr(self, path_type + '_acceleration')
        direction_change = (getattr(self, path_type + '_direction_change') / 360.0) * 2 * PI
        # Probability of [no change; full stop; direction adjust; speed adjust]
        move_probabilities = getattr(self, path_type + '_move_proba')
        assert np.isclose(np.sum(move_probabilities), 1.0), "The probabilities must sum to 1, "\
            "got {}".format(np.sum(move_probabilities))

        number_paths = int(
            len(self.valid_positions) * self.moving_paths_multiplier * undersample
        )
        x_values = [position[0] for position in self.valid_positions]
        y_values = [position[1] for position in self.valid_positions]
        x_range = [min(x_values), max(x_values)]
        y_range = [min(y_values), max(y_values)]
        del x_values, y_values

        moving_paths = []
        tqdm_str = "Computing {} paths...".format("pedestrian" if path_type == 'p' else "car")
        for _ in tqdm(range(number_paths), desc=tqdm_str, total=number_paths):
            valid_path = False
            while not valid_path:
                # Initializes the speed as 0, and its position as in one of the valid_positions
                this_speed_x = 0.0
                this_speed_y = 0.0
                this_x, this_y = list(self.valid_positions.keys())[
                    np.random.choice(len(self.valid_positions))
                ]
                this_sequence_indexes = [
                    self.valid_positions[(round(this_x), round(this_y))]
                ]
                movement_type = None
                direction = None

                # Loop until we have a path with length = `time_steps`
                for _ in range(self.time_steps - 1):
                    #-------------------------------------------------------------
                    # Updates the pedestrian direction and speed
                    curr_speed = np.sqrt(np.square(this_speed_x) + np.square(this_speed_y))

                    # After full stop, restarts in random direction with avg speed
                    if curr_speed == 0.0:
                        direction = np.random.uniform(0, 2 * PI)
                        this_speed_x = avg_speed * np.cos(direction)
                        this_speed_y = avg_speed * np.sin(direction)
                    # Full stop
                    elif movement_type == 1:
                        this_speed_x = 0.0
                        this_speed_y = 0.0
                    # Direction adjust
                    elif movement_type == 2:
                        direction_adjust = np.random.uniform(-direction_change,
                            direction_change)
                        direction += direction_adjust
                        this_speed_x = curr_speed * np.cos(direction)
                        this_speed_y = curr_speed * np.sin(direction)
                    # Speed adjust
                    elif movement_type == 3:
                        speed_adjust = np.random.uniform(-acceleration,
                            acceleration)
                        curr_speed += speed_adjust
                        curr_speed = np.clip(curr_speed, 0.0, max_speed)
                        this_speed_x = avg_speed * np.cos(direction)
                        this_speed_y = avg_speed * np.sin(direction)
                    # Otherwise, must be a "no_change" movement type
                    else:
                        assert movement_type == 0, "If the code reached this line, the movement "\
                            "type must be a 'no_change'."

                    #-------------------------------------------------------------
                    # Given the speed/direction, updates position
                    # [positions in meters, speed in meters per second, 1 sample per second]
                    this_x += this_speed_x
                    this_y += this_speed_y

                    #-------------------------------------------------------------
                    # Checks if position is valid: if it is, stores it, otherwise restarts the path
                    x_index, y_index = round(this_x), round(this_y)
                    if not x_range[0] < x_index < x_range[1]:
                        break
                    elif not y_range[0] < y_index < y_range[1]:
                        break
                    elif (x_index, y_index) not in self.valid_positions:
                        break
                    else:
                        this_touple = (x_index, y_index)
                        this_sequence_indexes.append(self.valid_positions[this_touple])

                    #-------------------------------------------------------------
                    # Selects next movement type for the next time step
                    if len(this_sequence_indexes) == self.time_steps:
                        valid_path = True
                    movement_type = np.random.choice(4, p=move_probabilities)

            assert len(this_sequence_indexes) == self.time_steps
            moving_paths.append(this_sequence_indexes)
        return moving_paths

    def _create_path_set(self, undersample=1.):
        """ Create a dictionary of paths for a specific set, given the loaded options.
        :param undersample: floating point between 0 and 1 that undersamples the dictionary of
            paths, based on this ratio
        :returns: a dictionary containing lists of paths, per path type
        """
        # Initializes the paths as a dictionary of empty lists, with a field per
        # path type ('s' for static, 'p' for pedestrian, and 'c' for car)
        paths = {'s': [], 'p': [], 'c': []}
        number_of_paths = 0
        assert 0. <= undersample <= 1., "`undersample` must be a floating point number between 0 "\
            "and 1, got {}".format(undersample)

        # Processes the static paths (just stores the valid locations)
        if self.s_paths:
            paths['s'] = self._get_static_paths(undersample)
            number_of_paths += len(paths['s'])

        # Processes the pedestrian paths
        if self.p_paths:
            paths['p'] = self._get_moving_paths('p', undersample)
            number_of_paths += len(paths['p'])

        # Processes the car paths
        if self.c_paths:
            paths['c'] = self._get_moving_paths('c', undersample)
            number_of_paths += len(paths['c'])

        logging.info("Created a total of %s paths", number_of_paths)
        return paths

    def create_paths(self):
        """ Creates three distinct path sets (train/validation/test)
        """
        self._paths = {}
        logging.info("Creating paths for the train set...")
        self._paths["train"] = self._create_path_set()
        logging.info("Creating paths for the validation set...")
        self._paths["validation"] = self._create_path_set(undersample=0.1)
        logging.info("Creating paths for the train set...")
        self._paths["test"] = self._create_path_set()

    def check_existing_paths(self):
        """ Checks whether the paths we are trying to create already exists

        :returns: Boolean flag, with `True` meaning that the paths already exists
        """
        paths_exist = False
        if os.path.isfile(self.paths_file):
            with open(self.paths_file, 'rb') as paths_file:
                _, target_paths_id = pickle.load(paths_file)
            if target_paths_id == self.paths_id:
                paths_exist = True
        return paths_exist

    def store_paths(self):
        """ Stores the created paths
        """
        target_folder = os.path.split(self.paths_file)[0]
        if not os.path.exists(target_folder):
            logging.info("Target folder (%s) not found, creating it...", target_folder)
            os.makedirs(target_folder)
        logging.info("Storing the result ...")
        with open(self.paths_file, 'wb') as data_file:
            pickle.dump([self._paths, self.paths_id], data_file)

    def load_paths(self):
        """ Loads the previously stored paths, returning them

        :returns: previously stored paths
        """
        assert self.check_existing_paths(), "The paths with the specified path ({}) either "\
            "do not exists or were built with different simulation settings. Please run the "\
            "data preprocessing step with the new simulation settings!".format(
            self.paths_file)

        with open(self.paths_file, 'rb') as data_file:
            paths, _ = pickle.load(data_file)
        return paths


def get_paths_id(data_settings, path_settings):
    """ Creates and returns an unique ID (for practical purposes), given the data/path
    parameters. The main use of this ID is to make sure we are using the correct data source,
    and that the data/path parameters weren't changed halfway through the simulation sequence.
    """
    hashing_features = [
        data_settings['pos_grid'],
        path_settings['time_steps'],
        path_settings['moving_paths_multiplier'],
        path_settings['s_paths'],
        path_settings['p_paths'],
        path_settings['p_avg_speed'],
        path_settings['p_max_speed'],
        path_settings['p_acceleration'],
        path_settings['p_direction_change'],
        path_settings['p_move_proba'],
        path_settings['c_paths'],
        path_settings['c_avg_speed'],
        path_settings['c_max_speed'],
        path_settings['c_acceleration'],
        path_settings['c_direction_change'],
        path_settings['c_move_proba'],
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
