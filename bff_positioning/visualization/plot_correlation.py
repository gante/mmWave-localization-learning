""" Script to plot X/Y correlation from MC Dropout

(Instructions to run: `python path/to/plot_correlation.py <path to .yaml file>`)
"""

#pylint: disable=wrong-import-position

import os
import sys
import copy
import pickle
import logging
import yaml

from tqdm import tqdm
import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


GRID_SIZE = 400
PLOT_PATH = "correlation.jpg"
MAX_CORR = 1.
MAP_ALPHA = 0.5


def plot_correlation(y_true, y_pred, background_data):
    """ Plots the correlation of the MC Dropout samples

    :param y_true: ground truth
    :param y_pred: model samples
    :param background_data: numpy array containing the positions with data
    """

    correlations = np.zeros((int(GRID_SIZE)+1, int(GRID_SIZE)+1))

    # Computes correlations, scaling up the [0, 1] labels
    # NOTE: the correlation signal is inverted becaiuse `imshow` will invert the y axis
    for idx in tqdm(range(y_true.shape[0]), desc="Computing correlations"):
        x = round(y_true[idx, 0] * GRID_SIZE)
        y = round(y_true[idx, 1] * GRID_SIZE)
        correlations[x, y] = -scipy.stats.pearsonr(y_pred[idx, 0, :], y_pred[idx, 1, :])[0]

    # flips Y axis to produce the correct img, because imshow is derp
    correlations = np.flip(np.transpose(correlations), 0)
    logging.info("Min correlation %s at %s", np.min(correlations),
        np.unravel_index(correlations.argmin(), correlations.shape))
    logging.info("Max correlation %s at %s", np.max(correlations),
        np.unravel_index(correlations.argmax(), correlations.shape))

    # Plots the correlations
    fig = plt.figure(1, figsize=[7, 6])
    ax = fig.add_subplot(111)
    cax = plt.imshow(
        correlations,
        vmin=-MAX_CORR,
        vmax=MAX_CORR,
        cmap="coolwarm"
    )
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    ticks = np.linspace(-MAX_CORR, MAX_CORR, 11, dtype=float)
    cbar = plt.colorbar(cax, ticks=ticks)
    tick_labels = [str(round(value, 2)) for value in ticks]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.set_ylabel('Pearson Correlation Coefficient')

    # Plots the background on top
    my_cmap = copy.copy(plt.cm.get_cmap('Greys')) # get a copy of the gray color map
    my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
    background_data[background_data > 0.5] = np.nan
    background_data[background_data < 0.5] = 0.5
    ax.imshow(-background_data, cmap=my_cmap, vmin=-1.0, vmax=0.0)

    plt.savefig(PLOT_PATH, dpi=300)
    logging.info("Plot written to %s", PLOT_PATH)


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
    experiment_settings = experiment_config['experiment_settings']
    ml_parameters = experiment_config["ml_parameters"]

    background_data_path = os.path.join(
        os.path.split(data_parameters["preprocessed_file"])[0], "existing_data_points.npy"
    )
    background_data = np.load(background_data_path)

    # Loads the predictions
    logging.info("Loading stored predictions...")
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    preditions_file = os.path.join(
        ml_parameters["model_folder"],
        experiment_name + '_' + experiment_settings["predictions_file"]
    )
    with open(preditions_file, 'rb') as pred_file:
        y_true, y_pred = pickle.load(pred_file)
    assert y_true.shape[0] == y_pred.shape[0], \
        "The loaded variables must have the same number of positions"
    assert y_true.shape[1] == y_pred.shape[1], \
        "The loaded variables must have the same physical dimensions!"
    assert len(y_true.shape) == 2, "The label data should only have two dimensions"
    assert y_pred.shape[2] > 1, "Not enough MC Dropout samples"
    logging.info("%s predictions loaded\n", y_true.shape[0])

    # Plots the uncertainty on the map
    plot_correlation(y_true, y_pred, background_data)


if __name__ == '__main__':
    main()
