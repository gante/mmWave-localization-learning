"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python run_non_tracking_experiment.py <path to .yaml file>`)
"""

import sys
import logging
import yaml

from bff_positioning.data import Preprocessor, create_noisy_features
from bff_positioning.models import CNN

def main():
    """Main block of code, which runs the experiment"""

    logging.basicConfig(level="INFO")

    # Load the .yaml data and unpacks it
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)
    simulation_settings = experiment_config['simulation_settings']
    data_parameters = experiment_config['data_parameters']
    ml_parameters = experiment_config['ml_parameters']

    # Initializes the model and prepares it for training
    if simulation_settings["model_type"] == "cnn":
        model = CNN(ml_parameters)
    else:
        raise ValueError("The simulation settings specified 'model_type'={}. Currently, only "
            "'cnn' is supported. [If you were looking for the HCNNs: sorry, the code was quite "
            "lengthy, so I moved its refactoring into a future to do. Please contact me if you "
            "want to experiment with it.]".format(simulation_settings["model_type"]))
    model.set_graph()

    # Loads the dataset
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()

    # Creates the test set
    # features_test, labels_test = create_noisy_features(features, labels,
                        # noise_std_converted, min_pow_cutoff, scaler, only_16_bf)

    # Clean up
    model.close()

if __name__ == '__main__':
    main()
