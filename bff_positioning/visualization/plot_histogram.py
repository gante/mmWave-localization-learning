""" Script to plot the cumulative histogram for the predictions error

(Instructions to run: `python path/to/plot_histogram.py <path to .yaml file>`)
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

HISTOGRAM_BINS = 1000
GRID_SIZE = 400
PLOT_PATH = "hist.pdf"

def plot_custom_cumulative_histogram(sorted_distances, n_bins):
    """ Plots and stores a custom cumulative histogram, as used in the papers

    :param sorted_distances: numpy array with the sorted distances
    """
    # Computes the histogram bins
    len_predictions = sorted_distances.shape[0]
    hist_bin = [0]*n_bins
    for idx in range(n_bins):
        current_percentile = idx/n_bins
        hist_bin[idx] = sorted_distances[int(len_predictions * current_percentile)]

    # Plots the full cumulative histogram
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.45, 0.2, 0.4, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])
    x = hist_bin
    y = np.linspace(0, 1, n_bins)
    ax1.plot(x, y)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, x[-1]])
    ax1.set_ylabel("Cumulative Histogram", fontsize=12)
    ax1.set_xlabel("Error (m)", fontsize=12)

    # Mark 95%
    distance_95 = sorted_distances[int(0.95 * len_predictions)]
    ax1.plot([0, distance_95], [0.95, 0.95], color='tab:orange', linestyle=':')
    ax1.plot([distance_95, distance_95], [0.0, 0.09], color='tab:orange', linestyle=':')
    ax1.plot([distance_95, distance_95], [0.19, 0.95], color='tab:orange', linestyle=':')
    special_annotation = "$95^{th}$"    #<-- the "th" is superscript
    ax1.annotate(
        special_annotation + " percentile:{:.4f}\n".format(distance_95),
        xy=(distance_95+1, 0),
        xytext=(distance_95-10, 0.1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    # Creates a sub graph: cumulative histogram excluding the last 10%
    limit_idx = int(n_bins * 0.9)
    ax2.plot(x[:limit_idx], y[:limit_idx])
    ax2.set_xlim([0, x[limit_idx]])
    ax2.set_ylim([0, 0.9])
    ax2.set_title("Zoomed cumulative histogram,\n excluding the last 10%")

    # Mark 50% in the subgraph
    distance_50 = sorted_distances[int(0.50 * len_predictions)]
    ax2.plot([0, distance_50], [0.50, 0.50], color='tab:orange', linestyle=':')
    ax2.plot([distance_50, distance_50], [0.0, 0.50], color='tab:orange', linestyle=':')
    ax2.annotate(
        "Median:{:.4f}\n".format(distance_50),
        xy=(distance_50+0.1, 0),
        xytext=(distance_50+1, 0.15),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

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
    logging.info("%s predictions loaded\n", y_true.shape[0])

    # Gets an array with the sorted distances and prints the RMSE
    logging.info("Computing distances and the RMSE...")
    sorted_distances = np.sort(np.sqrt(np.sum(np.square(y_true - y_pred), 1))) * GRID_SIZE
    rmse = np.sqrt(np.mean(np.square(sorted_distances)))
    logging.info("RMSE = %.4f\n", rmse)

    # Getting the cumulative histogram
    logging.info("Getting the cumulative histogram...")
    plot_custom_cumulative_histogram(sorted_distances, HISTOGRAM_BINS)

    # Prints a few final notes
    logging.info("NOTE - this script was built for the default data settings")

if __name__ == '__main__':
    main()
