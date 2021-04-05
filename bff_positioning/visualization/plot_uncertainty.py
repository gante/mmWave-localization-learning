""" Script to plot samples from MC Dropout

(Instructions to run: `python path/to/plot_uncertainty.py <path to .yaml file>`)
"""

#pylint: disable=wrong-import-position

import os
import sys
import pickle
import logging
import yaml
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, BboxPatch


ZOOM_SIZE = 15 # +/-, in meters
N_SAMPLES = 10
GRID_SIZE = 400
PLOT_PATH = "uncertainty.jpg"
MAP_ALPHA = 0.5


def plot_uncertainty(y_true, y_pred, background_data):
    """ Plots the samples over background data

    :param y_true: ground truth
    :param y_pred: model samples
    :param background_data: numpy array containing the positions with data
    """
    _, ax = plt.subplots()

    # Flips y (imshow assumes 0 on top, growing down)
    y_true[:, 1] = -y_true[:, 1] + GRID_SIZE
    y_pred[:, 1] = -y_pred[:, 1] + GRID_SIZE

    # Plots the background in black and black (white = positions with data)
    ax.imshow(-background_data, cmap='Greys', vmin=-1.0, vmax=0.0, alpha=MAP_ALPHA)

    # Plots the MC Dropout samples in blue, with some transparency
    ax.scatter(
        x=y_pred[:, 0, :],
        y=y_pred[:, 1, :],
        c='b', s=3, alpha=0.15, edgecolors='none',
        label="MC Dropout samples"
    )

    # Plots the true position in solid red
    ax.scatter(
        x=y_true[:, 0],
        y=y_true[:, 1],
        c='r', s=5, edgecolors='none',
        label="True positions"
    )
    ax.legend(loc="upper right", labelcolor=['b', 'r'], fontsize="small")

    # Plots a zoomed detail
    axins = zoomed_inset_axes(ax, zoom=3, loc='lower left')
    axins.imshow(-background_data, cmap='Greys', vmin=-1.0, vmax=0.0, alpha=MAP_ALPHA)
    rand_idx = np.random.randint(low=0, high=y_true.shape[0], size=1)
    axins.scatter(
        x=y_pred[rand_idx, 0, :],
        y=y_pred[rand_idx, 1, :],
        c='b', s=4, alpha=0.15, edgecolors='none',
    )
    axins.scatter(x=y_true[rand_idx, 0], y=y_true[rand_idx, 1], c='r', edgecolors='none')

    # Limit the region for zoom
    axins.set_xlim(y_true[rand_idx, 0]-ZOOM_SIZE, y_true[rand_idx, 0]+ZOOM_SIZE)
    axins.set_ylim(y_true[rand_idx, 1]-ZOOM_SIZE, y_true[rand_idx, 1]+ZOOM_SIZE)

    # Hides ticks
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and connecting lines between
    # the bbox and the inset axes area. Also inverts the y of the inner plot, as the outer plot
    # is inverted (imshow)
    axins.invert_yaxis()

    # ================================================================
    # (adapted from `mark_inset`, removing the connectors)
    parent_axes = ax
    inset_axes = axins
    kwargs = {"fc": "none", "ec": "0.0"}
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
    pp = BboxPatch(rect, fill=fill, **kwargs)
    parent_axes.add_patch(pp)
    # ================================================================

    plt.draw()
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

    # Pick a random positions
    selected_positions = np.sort(np.random.randint(low=0, high=y_true.shape[0], size=N_SAMPLES))
    y_true_position = y_true[selected_positions, :] * GRID_SIZE
    y_pred_position = y_pred[selected_positions, ...] * GRID_SIZE
    logging.info("Selected positions (labels): \n%s", y_true_position)

    # Plots the uncertainty on the map
    plot_uncertainty(y_true_position, y_pred_position, background_data)


if __name__ == '__main__':
    main()
