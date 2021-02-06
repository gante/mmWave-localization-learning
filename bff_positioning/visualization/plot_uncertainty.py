""" Script to plot samples from MC Dropout

(Instructions to run: `python path/to/plot_uncertainty.py <path to .yaml file>`)

WORK IN PROGRESS, requires manually saving model data as a numpy array.
Expected input format: 2D array with MC Dropout samples from the model, 1D array with true position
"""

#pylint: disable=wrong-import-position

import os
import sys
import logging
import yaml
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


PLOT_PATH = "uncertainty.pdf"


def plot_uncertainty(y_true, y_pred, background_data):
    """ Plots the samples over background data

    :param y_true: ground truth
    :param y_pred: model samples
    :param background_data: numpy array containing the positions with data
    """
    # Plots the background in black and black (white = positions with data)
    plt.imshow(-background_data, cmap='Greys', vmin=-1.0, vmax=0.0, alpha=0.5)

    # Plots the MC Dropout samples in blue, with some transparency
    plt.scatter(x=y_pred[:, 0], y=y_pred[:, 1], c='b', s=2, alpha=0.1)

    # Plots the true position in solid red
    plt.scatter(x=y_true[0], y=y_true[1], c='r', s=5)

    # Saves the plot
    plt.savefig(PLOT_PATH, format='pdf')
    logging.info("Plot written to %s", PLOT_PATH)


def main():
    """Main block of code, which controls the plotting"""

    # For the final version:
    # 1 - Make a new script, `sample_model`, to store MC Dropout output data
    # 2 - Make this script load those samples (copy logic from other plot scripts)
    # 3 - Pick randomly N positions and store them as basename_X_Y.pdf

    logging.basicConfig(level="INFO")

    # Load the .yaml data and unpacks it
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)
    data_parameters = experiment_config['data_parameters']

    background_data_path = os.path.join(
        os.path.split(data_parameters["preprocessed_file"])[0], "existing_data_points.npy"
    )
    background_data = np.load(background_data_path)

    # REPLACE THIS WITH ACTUAL DATA <======================================================================================
    # (remember that the model has the output range in [0, 1], and it has to be scaled to [0, 400])
    # 1D array with true position
    y_true = np.asarray([200, 200])
    # 2D array with MC Dropout samples from the model
    y_pred = np.random.normal(loc=200, scale=10, size=(1000, 2))

    # Plots the uncertainty on the map
    plot_uncertainty(y_true, y_pred, background_data)


if __name__ == '__main__':
    main()
