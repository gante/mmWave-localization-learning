""" Script to plot basic dataset information: maximum received power AND number of dictinct
paths at each position

(Instructions to run: `python path/to/plot_dataset.py <path to .yaml file>`)
"""

#pylint: disable=wrong-import-position

import sys
import logging
import yaml
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from bff_positioning.data import Preprocessor


GRID_SIZE = 400.
MIN_POWER = -190.
MAX_POWER = -40.
MIN_PATHS = 0
MAX_PATHS = 500
PLOT_POWER_PATH = "power_map.png"
PLOT_PATHS_PATH = "paths_map.png"


def plot_dataset(features, labels):
    """ Plots basic dataset info (maximum received power AND number of dictinct
    paths at each position)

    :param features: numpy array with the features (in dBm)
    :param labels: numpy array with the labels (in m)
    """

    position_power = np.full((int(GRID_SIZE)+1, int(GRID_SIZE)+1), MIN_POWER)
    position_paths = np.zeros((int(GRID_SIZE)+1, int(GRID_SIZE)+1))

    for idx in range(features.shape[0]):
        x = round(labels[idx, 0])
        y = round(labels[idx, 1])
        position_paths[x, y] = np.count_nonzero(features[idx, :] > MIN_POWER)
        position_power[x, y] = np.max(features[idx, :])

    # Plots the thing (flips Y axis to produce the correct img, because imshow is derp)
    logging.info("Plotting power...")
    fig = plt.figure(1, figsize=[7, 6])
    ax = fig.add_subplot(111)
    half_grid = int(GRID_SIZE/2)
    im_extent = [-half_grid, half_grid, -half_grid, half_grid]
    cax = plt.imshow(
        np.flip(np.transpose(position_power), 0),
        vmin=MIN_POWER,
        vmax=MAX_POWER,
        extent=im_extent
    )
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    ax.plot(0, 3, 'r^')
    ticks = np.linspace(MIN_POWER, MAX_POWER, 6, dtype=int)
    cbar = plt.colorbar(cax, ticks=ticks)
    tick_labels = [str(value) for value in ticks]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.set_ylabel('Maximum Received Power (dBm)')
    plt.savefig(PLOT_POWER_PATH)
    logging.info("Plot written to %s", PLOT_POWER_PATH)

    logging.info("Plotting paths...")
    fig = plt.figure(2, figsize=[7, 6])
    ax = fig.add_subplot(111)
    half_grid = int(GRID_SIZE/2)
    im_extent = [-half_grid, half_grid, -half_grid, half_grid]
    cax = plt.imshow(
        np.flip(np.transpose(position_paths), 0),
        vmin=MIN_PATHS,
        vmax=MAX_PATHS,
        extent=im_extent
    )
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    ax.plot(0, 3, 'r^')
    ticks = np.linspace(MIN_PATHS, MAX_PATHS, 6, dtype=int)
    cbar = plt.colorbar(cax, ticks=ticks)
    tick_labels = [str(value) for value in ticks]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.set_ylabel('Number of Distinct Paths Between Tx and Rx')
    plt.savefig(PLOT_PATHS_PATH)
    logging.info("Plot written to %s", PLOT_PATHS_PATH)


def main():
    """Main block of code, which controls the plotting"""

    logging.basicConfig(level="INFO")

    # Load the .yaml data and unpacks it
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)
    data_parameters = experiment_config['data_parameters']

    # Loads the processed data
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset(power_in_dbm=True)
    features = np.asarray(features)
    labels = np.asarray(labels)
    assert features.shape[0] == labels.shape[0], \
        "The number of dataset rows must match the number of labels"
    assert labels.shape[1] == 2, "The labels must be vectors with length 2 (x and y position)"

    # Upsacles the labels to the original scale, sets power-less features to a low number
    labels *= GRID_SIZE
    features = np.where(features == 0.0, MIN_POWER, features)
    logging.info("Labels range (m): x=[%.2f, %.2f], y=[%.2f, %.2f]",
        np.min(labels[:, 0]), np.max(labels[:, 0]), np.min(labels[:, 1]), np.max(labels[:, 1]))
    logging.info("Features range (dBm):  [%.2f, %.2f]", np.min(features), np.max(features))

    # Plots the power on the map
    plot_dataset(features, labels)


if __name__ == '__main__':
    main()
