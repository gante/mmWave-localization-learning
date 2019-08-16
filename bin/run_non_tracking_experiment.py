"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python run_non_tracking_experiment.py <path to .yaml file>`)
"""

import sys
import logging
import yaml

from bff_positioning.data import Preprocessor

def main():
    """Main block of code, which runs the experiment"""

    logging.basicConfig(level="INFO")

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)

    # Checks if the desired pre-processed data exists
    data_preprocessor = Preprocessor(experiment_config['data_parameters'])
    dataset_exists = data_preprocessor.check_existing_dataset()
    if not dataset_exists:
        logging.error("The dataset with the specified path (%s) and/or simulation settings "
            "(defined in %s) does not exist. Please run the data preprocessing step with the"
            "the same simulation settings.", data_preprocessor.preprocessed_file, sys.argv[1])


if __name__ == '__main__':
    main()
