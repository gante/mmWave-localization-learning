"""
Preprocesses the dataset, preparing all data for training.

The arguments are loaded from a .yaml file, which is the input argument of this script
(Instructions to run: `python preprocess_dataset.py <path to .yaml file>`)
"""

import sys
import time
import logging
import yaml

from bff_positioning.data import Preprocessor, PathCreator

def main():
    """Main block of code, which runs the data-preprocessing"""

    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python preprocess_dataset.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.safe_load(yaml_config_file)
    data_parameters = experiment_config['data_parameters']
    path_parameters = experiment_config['path_parameters'] \
        if 'path_parameters' in experiment_config else None

    # Checks if the dataset we are trying to create already exists
    data_preprocessor = Preprocessor(data_parameters)
    dataset_exists = data_preprocessor.check_existing_dataset()

    # If it doesn't, creates it
    if not dataset_exists:
        # Run the data_preprocessor
        data_preprocessor.create_bff_dataset()

        # Stores the processed data and a id. That id is based on the simulation
        # settings for the preprocessing part, and it's used to make sure future uses
        # of this preprocessed data match the desired simulation settings
        data_preprocessor.store_dataset()
    else:
        logging.info("The dataset already exists in %s, skipping the dataset creation "
            "steps!", data_parameters['preprocessed_file'])

    # If the experiment specifies the `path_parameters` field (i.e. a tracking experiment),
    # creates the paths. The overall flow is the same as above.
    if path_parameters:
        _, labels = data_preprocessor.load_dataset()
        path_creator = PathCreator(data_parameters, path_parameters, labels)
        paths_exist = path_creator.check_existing_paths()
        if not paths_exist:
            path_creator.create_paths()
            path_creator.store_paths()
        else:
            logging.info("The paths already exist in %s, skipping their creation!",
                data_parameters['paths_file'])

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)


if __name__ == '__main__':
    main()
