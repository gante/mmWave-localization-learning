"""
Evaluates a model performance (throughput and power) on an Nvidia Jetson board.
Assumes that the model folder has been copied to a local folder on the board, and then
evaluates it by injecting random input data.

The arguments are loaded from a .yaml file, which is the input argument of this script
(Instructions to run: `sudo python jetson_performance.py <path to .yaml file>`)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging
import yaml
import numpy as np
from tqdm import tqdm

from bff_positioning.models import CNN, LSTM, TCN

# 32 = batch size; total samples >> batch size, to avoid cached results
TEST_SAMPLES = 32 * 64
# Must be enough to keep the GPU going for at least 1 minute (30 sec warm up, 30 sec measurement)
LOOP_SIZE = 100
MONITOR_RESULTS = os.path.expanduser("~/monitor_results.txt")


def main():
    """Main block of code, which runs the performance evaluation"""

    logging.basicConfig(level="INFO")
    logging.warning("This script should be run with SUDO on a jetson device! "
        "(CUDA might not be accessible otherwise)")

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


    # Creates dummy input data, prepares the monitor subprocess
    features_dummy = np.random.random_sample(tuple([TEST_SAMPLES] + model.input_shape))
    # monitor path, relative to this file: /jetson_power_monitor.sh
    jetson_monitor_path = os.path.join(
        str(Path(__file__).parent.absolute()), # parent to get ".."
        "jetson_power_monitor.sh"
    )
    subprocess.run(["chmod", "+x", jetson_monitor_path])
    assert not os.path.exists(MONITOR_RESULTS), "A monitor results file ({}) already exists. "\
        "Please delete or move it, and run this script again".format(MONITOR_RESULTS)

    # Prediction loop
    for i in tqdm(range(LOOP_SIZE)):
        if i == np.floor(LOOP_SIZE/2):
            subprocess.Popen([jetson_monitor_path])
            start = time.time()
        model.predict(features_dummy)

    # Checks if the monitor file was created
    end = time.time()
    assert os.path.exists(MONITOR_RESULTS), "The monitor didn't finish running, which means "\
        "the prediction is too short. Please increase `LOOP_SIZE`."
    logging.info("Power-related information stored in %s", MONITOR_RESULTS)

    # Prints prediction time
    exec_time = (end-start)
    logging.info("Total monitored prediction time: %.5E seconds", exec_time)
    samples_per_sec = TEST_SAMPLES*np.floor(LOOP_SIZE/2)/exec_time
    logging.info("Samples predicted per second: %s", samples_per_sec)
    logging.info("Samples predicted during monitoring: %s", samples_per_sec*30.)


if __name__ == '__main__':
    main()
