"""
Evaluates a model performance (throughput and power) on an Nvidia Jetson board.
Assumes that the model folder has been copied to a local folder on the board, and then
evaluates it by injecting random input data.

(Instructions to run: `python jetson_performance.py <path to model folder>`)
"""

import os
import sys
import time
import logging


# For more details, check the "Thermal Design Guide" provided by Nvidia
GPU_RAIL = ("VDD_SYS_GPU", "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input")


def main():
    """Main block of code, which runs the performance evaluation"""

    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data
    # assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
    #     "passed as a positional argument to this script. \n\n"\
    #     "E.g. `python preprocess_dataset.py <path to .yaml file>`"
    # with open(sys.argv[1], "r") as yaml_config_file:
    #     logging.info("Loading simulation settings from %s", sys.argv[1])
    #     experiment_config = yaml.safe_load(yaml_config_file)
    # data_parameters = experiment_config['data_parameters']
    # path_parameters = experiment_config['path_parameters'] \
    #     if 'path_parameters' in experiment_config else None

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)


if __name__ == '__main__':
    main()
