"""
Preprocesses the dataset.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python3 preprocess_dataset.py <path to .yaml file>`)
"""

import sys
import yaml

from bff_positioning.data_ops import Preprocessor

def main():
    """Main block of code, which runs the experiment"""

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python3 preprocess_dataset.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        print("Loading simulation settings from {}".format(sys.argv[1]))
        experiment_config = yaml.load(yaml_config_file)

    # Run the data_preprocessor
    data_preprocessor = Preprocessor(experiment_config)
    data_preprocessor()
    del data_preprocessor

    # Stores the processed data and a `sha_id`. That id is based on the simulation
    # settings for the preprocessing part, and it's used to make sure future uses
    # of this preprocessed data match the desired simulation settings


if __name__ == '__main__':
    main()
