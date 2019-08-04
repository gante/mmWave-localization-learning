"""
Preprocesses the dataset.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python preprocess_dataset.py <path to .yaml file>`)
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
        "E.g. `python preprocess_dataset.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)

    # Checks if the dataset we are trying to create already exists
    data_preprocessor = Preprocessor(experiment_config['data_parameters'])
    dataset_exists = data_preprocessor.check_existing_dataset()

    # If it doesn't, creates it
    if not dataset_exists:
        # Run the data_preprocessor
        data_preprocessor.create_bff_dataset()

        # Stores the processed data and a id. That id is based on the simulation
        # settings for the preprocessing part, and it's used to make sure future uses
        # of this preprocessed data match the desired simulation settings
        data_preprocessor.store_dataset()


if __name__ == '__main__':
    main()
