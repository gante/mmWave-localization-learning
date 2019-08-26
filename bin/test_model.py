"""
Runs the test dataset over the data, and stores the predictions

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python test_model.py <path to .yaml file>`)
"""

import os
import sys
import logging
import yaml

from bff_positioning.data import Preprocessor, create_noisy_features, get_95th_percentile
from bff_positioning.models import CNN

TESTS_PER_POSITION = 30


def main():
    """Main block of code, which runs the tests"""

    logging.basicConfig(level="INFO")

    # Load the .yaml data and unpacks it
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)
    experiment_settings = experiment_config['experiment_settings']
    data_parameters = experiment_config['data_parameters']
    ml_parameters = experiment_config['ml_parameters']

    # Loads the raw dataset
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()

    # Initializes the model and loads it
    logging.info("Initializing the model...")
    if experiment_settings["model_type"] == "cnn":
        model = CNN(ml_parameters)
    else:
        raise ValueError("The simulation settings specified 'model_type'={}. Currently, only "
            "'cnn' is supported. [If you were looking for the HCNNs: sorry, the code was quite "
            "lengthy, so I moved its refactoring into a future to do. Please contact me if you "
            "want to experiment with it.]".format(experiment_settings["model_type"]))
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    model.load(model_name=experiment_name)



    model.close()


if __name__ == '__main__':
    main()
