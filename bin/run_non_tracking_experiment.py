"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python3 run_non_tracking_experiment.py <path to .yaml file>`)
"""

import sys
import yaml

from mmwave_positioning.data_operations import Preprocessor

def main():
    """Main block of code, which runs the experiment"""

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python3 run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        experiment_config = yaml.load(yaml_config_file)

    # Run the data_preprocessor
    data_preprocessor = Preprocessor(experiment_config)
    data_preprocessor()
    del data_preprocessor


if __name__ == '__main__':
    main()
