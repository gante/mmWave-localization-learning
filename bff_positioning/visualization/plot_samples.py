""" Script to plot random data samples (the beamformed fingerprints)

(Instructions to run: `python path/to/plot_samples.py <path to .yaml file>`)
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


SAMPLES_PER_PLOT = 10
GRID_SIZE = 400
PLOT_PATH = 'bff_samples.pdf'


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

    # Loads the dataset
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()

    # Plots data
    logging.info("Plotting...")
    beamformings = data_parameters["beamformings"]
    time_slots = int(features.shape[1] / beamformings)
    to_plot = np.random.randint(features.shape[0], size=SAMPLES_PER_PLOT)
    plt.figure(1, figsize=[30, 5])     # figsize=[7, 4] for 3 images = plot in the papers
    for idx in range(SAMPLES_PER_PLOT):
        plt.subplot(1, SAMPLES_PER_PLOT, idx+1)
        bff_2d = features[to_plot[idx], ...].reshape(beamformings, time_slots)

        X = int((labels[to_plot[idx]][0] * GRID_SIZE) - (GRID_SIZE/2))
        Y = int((labels[to_plot[idx]][1] * GRID_SIZE) - (GRID_SIZE/2))
        title = 'X=' + str(X) + ' Y=' + str(Y)
        plt.title(title)

        plt.xlabel('Beamforming Index')
        if idx == 0:
            plt.ylabel('Time-Domain Sample Number')
        cax = plt.imshow(np.transpose(bff_2d), vmin=0.0, vmax=1.2)

    # Saves the plot
    cbar = plt.colorbar(cax, ticks=[0.0, 0.2, 0.7, 1.2])
    cbar.ax.set_yticklabels(['no_data', '-150dBm', '-100dBm', '-50dBm'])
    plt.savefig(PLOT_PATH, format='pdf')
    logging.info("Plot written to %s", PLOT_PATH)

    # Prints a few final notes
    logging.info("NOTE 1 - this script was built for the default data settings")
    logging.info("NOTE 2 - the position of the sample (X and Y) is relative to the BS")


if __name__ == '__main__':
    main()
