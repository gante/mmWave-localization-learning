"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python3 run_non_tracking_experiment.py <path to .yaml file>`)
"""

import sys
import yaml

def main():
    """Main block of code, which runs the experiment"""

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python3 run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        experiment_config = yaml.load(yaml_config_file)

    # Load the preprocessed data, and double-checks its `sha_id`


if __name__ == '__main__':
    main()
