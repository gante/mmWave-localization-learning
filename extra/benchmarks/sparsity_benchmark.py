"""
Python script with some sparsity-related benchmarks, using the data structures from this
repository -- the beamformed fingerprint. Other than the input path for the raw data source,
which can be passed as argument, this script is self-contained.

Operations benchmarked:
1 - Add noise to a sparse floating point vector
2 - Binarise a (noisy) sparse floating point vector, given a threshold

Example usage: `python sparsity_benchmark ~/bff_data/final_table`

NOTE -- the `final_table` file can be downloaded from
https://drive.google.com/drive/folders/1gfbZKCsq4D1tvPzPHLftWljsVaL2pjg_?usp=sharing
"""

import logging
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import Binarizer

from bff_positioning.data import Preprocessor
from bff_positioning.data.dataset_handling import _convert_power_variables


# Fixed parameters for the benchmark. Unless explicitly stated under *square brackets* OR you
# understand the implications of changing a variable, I'd suggest not to touch them, as it may
# break a processing step (and the logic of the data).
DATA_PARAMETERS = {
    "input_file": None,             # will be set from the script argument
    "preprocessed_file": None,      # will be set from the script argument
    "max_time": 6,                  # in microseconds
    "sample_freq": 20,              # in MHz
    "beamformings": 32,
    "original_tx_power": 30,        # in dBm
    "original_rx_gain": 0,          # in dBi
    "power_offset": 170,
    "power_scale": 0.01,
    "pos_grid": [400.0, 400.0],     # in meters
    "pos_shift": [183.0, 176.0],    # in meters
    "keep_timeslots": [1, 82],      # [Remove to increase sparsity, this line filters super sparse
                                    # time slots]
    "run_sanity_checks": True
}
EXPERIMENT_PARAMETERS = {
    "detection_threshold": -100,    #in dBm, depends on the sample frequency -> thermal noise
    "tx_power": 45,                 #in dBm [Decrease to increase sparsity on existing data
                                    # (and vice-versa, increase to decrease sparsity)]
    "rx_gain": 10,                  #in dBi [Exact same effect as above]
    "noise_std": 6,                 #in dB, Gaussian noise applied over dB values -> "log-normal"
    "scaler_type": "binarizer"
}


def preprocess_dataset(data_parameters):
    """ Creates the BFF dataset from raw data (adapted from bin/preprocess_dataset.py) """

    assert data_parameters["input_file"] is not None
    assert data_parameters["preprocessed_file"] is not None

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
    return data_preprocessor


def benchmark_noise(features, data_parameters, experiment_parameters):
    """ Benchmark 1 -- Add noise to a sparse floating point vector"""
    # Gets the noise parameters from the settings
    scaled_noise, _ = _convert_power_variables(
        experiment_parameters,
        data_parameters
    )

    # Runs the benchmark
    logging.info("Starting benchmark 1 -- Add noise to a sparse floating point vector")
    start = time.time()

    noise = np.random.normal(scale=scaled_noise, size=features.shape)
    noisy_features = features + noise

    end = time.time()
    exec_time = (end - start) * 1000
    logging.info("Benchmark 1 execution time: %.3f milliseconds", exec_time)
    return noisy_features


def benchmark_binarization(noisy_features, data_parameters, experiment_parameters):
    """ Benchmark 2 -- Binarise a (noisy) sparse floating point vector, given a threshold"""
    # Gets cutoff parameters from the settings
    _, scaled_cutoff = _convert_power_variables(
        experiment_parameters,
        data_parameters
    )
    scaler = scaler = Binarizer(threshold=scaled_cutoff, copy=False)

    # Runs the benchmark
    logging.info("Starting Benchmark 2 -- Binarise a (noisy) sparse floating point vector, "
        "given a threshold")
    start = time.time()

    noisy_features = scaler.fit_transform(noisy_features)
    noisy_features = noisy_features.astype(bool)

    end = time.time()
    exec_time = (end - start) * 1000
    logging.info("Benchmark 2 execution time: %.3f milliseconds", exec_time)


def main():
    """ Main block of code, which runs the benchmarks"""
    logging.basicConfig(level="INFO")
    assert len(sys.argv) == 2, "Exactly one positional argument (path to the raw dataset) is "\
        "needed. \n\nE.g. `python sparsity_benchmark ~/bff_data/final_table`"

    # Prepares data for the benchmark, may take a while
    data_parameters = DATA_PARAMETERS.copy()
    data_parameters["input_file"] = sys.argv[1]
    data_parameters["preprocessed_file"] = os.path.join(
        os.path.dirname(data_parameters["input_file"]),
        "preprocessed_dataset.pkl"
    )
    data_preprocessor = preprocess_dataset(data_parameters=data_parameters)

    # Note: the features here should be in range [0, ~1.2], according to the original experiments.
    # 0 corresponds to no data, everything else is linearly scaled from dB units.
    features, _ = data_preprocessor.load_dataset()

    logging.info("Starting benchmarks")
    noisy_features = benchmark_noise(
        features=features,
        data_parameters=data_parameters,
        experiment_parameters=EXPERIMENT_PARAMETERS
    )
    benchmark_binarization(
        noisy_features=noisy_features,
        data_parameters=data_parameters,
        experiment_parameters=EXPERIMENT_PARAMETERS
    )
    logging.info("Done")


if __name__ == '__main__':
    main()
