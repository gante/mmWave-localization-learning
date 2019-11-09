"""
Evaluates a model performance (throughput and power) on an Nvidia Jetson board.
Assumes that the model folder has been copied to a local folder on the board, and then
evaluates it by injecting random input data.

The arguments are loaded from a .yaml file, which is the input argument of this script
(Instructions to run: `python jetson_performance.py <path to .yaml file>`)
"""

import os
import sys
import time
import logging
import yaml

from bff_positioning.models import CNN, LSTM, TCN


# For more details, check the "Thermal Design Guide" provided by Nvidia
GPU_RAIL = ("VDD_SYS_GPU", "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input")
TEST_SAMPLES = 32 * 64


def main():
    """Main block of code, which runs the performance evaluation"""

    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python jetson_performance.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.safe_load(yaml_config_file)
    experiment_settings = experiment_config['experiment_settings']
    ml_parameters = experiment_config['ml_parameters']
    path_parameters = experiment_config['path_parameters'] \
        if 'path_parameters' in experiment_config else None

    # Initializes the model and prepares it for testing
    logging.info("Initializing the model (type = %s)...", experiment_settings["model_type"].lower())
    if experiment_settings["model_type"].lower() == "cnn":
        ml_parameters["input_type"] = "float"
        model = CNN(ml_parameters)
    elif experiment_settings["model_type"].lower() in ("lstm", "tcn"):
        assert path_parameters, "This model requires `paths_parameters`. See the example."
        assert path_parameters["time_steps"] == ml_parameters["input_shape"][0], "The ML model "\
            "first input dimention must match the length of the paths! (path length = {}, model)"\
            "input = {})".format(path_parameters["time_steps"], ml_parameters["input_shape"][0])
        ml_parameters["input_type"] = "bool"
        if experiment_settings["model_type"].lower() == "lstm":
            model = LSTM(ml_parameters)
        else:
            model = TCN(ml_parameters)
    else:
        raise ValueError("The simulation settings specified 'model_type'={}. Currently, only "
            "'cnn', 'lstm', and 'tcn' are supported.".format(experiment_settings["model_type"]))
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    model.load(model_name=experiment_name)

    # To do:
    # create random samples
    # loops predictions over random samples (100 times), printing power and time
    # print final results (total time, total samples, avg throughput, avg power, energy per sample)

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)


if __name__ == '__main__':
    main()
