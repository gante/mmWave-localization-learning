""" Script to plot the the average error per position

(Instructions to run: `python path/to/plot_error_map.py <path to .yaml file>`)
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


GRID_SIZE = 400.
MAX_ERROR = 10
PLOT_PATH = "error_map.pdf"


def plot_error_map(y_true, y_pred, max_error):
    """ Plots the error on the map, with a color-code that depends on `max_error`

    :param y_true: ground truth
    :param y_pred: model predictions
    :param max_error: The upper limit of the color-code scale
    """
    distance_error = np.sqrt(np.sum(np.square(y_true - y_pred), 1))

    # Creates empty data structures
    position_entries = {
        (x, y): [] for x in range(int(GRID_SIZE)+1) for y in range(int(GRID_SIZE)+1)
    }
    position_error = np.full((int(GRID_SIZE)+1, int(GRID_SIZE)+1), 100)

    # For each prediction: gets the true position -> stores the error
    logging.info("Sorting the predictions by position...")
    for idx in range(y_true.shape[0]):
        x = round(y_true[idx, 0])
        y = round(y_true[idx, 1])
        position_entries[(x, y)].append(distance_error[idx])

    # Gets each position's average   (if it has no entries, sets as default value)
    logging.info("Averaging the results by position...")
    for x in range(int(GRID_SIZE)+1):
        for y in range(int(GRID_SIZE)+1):
            if position_entries[(x, y)]:
                position_error[x, y] = np.mean(position_entries[(x, y)])

    # Plots the thing (flips Y axis to produce the correct img, because imshow is derp)
    logging.info("Plotting...")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    half_grid = int(GRID_SIZE/2)
    im_extent = [-half_grid, half_grid, -half_grid, half_grid]
    cax = plt.imshow(np.flip(np.transpose(position_error), 0), vmin=0.0, vmax=10, extent=im_extent)
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    ax.plot(0,3,'r^')
    ticks = np.linspace(0, max_error, 6)
    cbar = plt.colorbar(cax, ticks=ticks)
    tick_labels = [str(value) for value in ticks]
    tick_labels[-1] += '+'
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Average Error (m)', rotation=270)

    # Saves the plot
    plt.savefig(PLOT_PATH, format='pdf')
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
    experiment_settings = experiment_config['experiment_settings']
    ml_parameters = experiment_config["ml_parameters"]

    # Loads the predictions
    logging.info("Loading stored predictions...")
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    preditions_file = os.path.join(
        ml_parameters["model_folder"],
        experiment_name + '_' + experiment_settings["predictions_file"]
    )
    with open(preditions_file, 'rb') as pred_file:
        y_true, y_pred = pickle.load(pred_file)
    assert y_true.shape == y_pred.shape, "The loaded variables must have the same shape!"
    logging.info("%s predictions loaded", y_true.shape[0])

    # Upsacles the predictions and the ground truth to the original scale
    y_true *= GRID_SIZE
    y_pred *= GRID_SIZE
    logging.info("Ground truth range: x=[%.2f, %.2f], y=[%.2f, %.2f]",
        np.min(y_true[:, 0]), np.max(y_true[:, 0]), np.min(y_true[:, 1]), np.max(y_true[:, 1]))
    logging.info("Predictions range:  x=[%.2f, %.2f], y=[%.2f, %.2f]",
        np.min(y_pred[:, 0]), np.max(y_pred[:, 0]), np.min(y_pred[:, 1]), np.max(y_pred[:, 1]))
    logging.info("Distinct ground truth values: x=%s, y=%s",
        np.unique(y_true[:, 0]).shape[0], np.unique(y_true[:, 1]).shape[0])

    # Plots the error on the map
    plot_error_map(y_true, y_pred, MAX_ERROR)

    # Prints a few final notes
    logging.info("NOTE - this script was built for the default data settings")

if __name__ == '__main__':
    main()
