"""
Runs the test dataset over the data, and stores the predictions

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python test_model.py <path to .yaml file>`)
"""

import os
import sys
import time
import logging
import pickle
import yaml
import numpy as np

from bff_positioning.data import Preprocessor, create_noisy_features, get_95th_percentile,\
    undersample_bf, undersample_space
from bff_positioning.models import CNN, score_predictions


def main():
    """Main block of code, which runs the tests"""

    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data and unpacks it
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python run_non_tracking_experiment.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.load(yaml_config_file)
    experiment_settings = experiment_config['experiment_settings']
    data_parameters = experiment_config['data_parameters']
    ml_parameters = experiment_config['ml_parameters']

    # Loads the raw dataset
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()

    # Undersamples the dataset (if requested)
    if "undersample_bf" in experiment_settings and experiment_settings["undersample_bf"]:
        features = undersample_bf(features, data_parameters["beamformings"])
    if "undersample_space" in experiment_settings:
        features, labels = undersample_space(features, labels, data_parameters["undersample_space"])

    # Initializes the model and loads it
    logging.info("Initializing the model...")
    if experiment_settings["model_type"] == "cnn":
        model = CNN(ml_parameters)
    else:
        raise ValueError("The simulation settings specified 'model_type'={}. Currently, only "
            "'cnn' is supported. [If you were looking for the HCNNs: sorry, the code was quite "
            "lengthy, so I moved its refactoring into a future to do. Please contact me if you "
            "want to experiment with it.]".format(experiment_settings["model_type"]))
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    model.load(model_name=experiment_name)

    # Prediction loop
    tests_per_position = experiment_settings["tests_per_position"]
    y_true = np.asarray([])
    y_pred = np.asarray([])
    for set_idx in range(tests_per_position):
        logging.info("Creating test set %2s out of %2s...", set_idx+1, tests_per_position)
        features_test, labels_test = create_noisy_features(
            features,
            labels,
            experiment_settings,
            data_parameters,
        )
        logging.info("Running predictions and storing data...\n")
        predictions_test = model.predict(features_test)
        y_true = np.vstack(y_true, labels_test)
        y_pred = np.vstack(y_pred, predictions_test)
        assert labels_test.shape[1] == y_true.shape[1], "The number of dimensions per sample "\
            "must stay constant!"
        assert y_true.shape == y_pred.shape, "The predictions and the labels must have the "\
            "same shape!"

    # Closes the model, gets the test scores, and stores predictions-labels pairs
    model.close()
    logging.info("Computing test metrics...")
    test_score = score_predictions(y_true, y_pred, ml_parameters["validation_metric"])
    test_score *= data_parameters["pos_grid"][0]
    test_95_perc = get_95th_percentile(
        y_true,
        y_pred,
        rescale_factor=data_parameters["pos_grid"][0]
    )
    logging.info("Average test distance: %.5f m || 95th percentile:  %.5f m\n",
        test_score, test_95_perc)
    preditions_file = os.path.join(
        ml_parameters["model_folder"],
        experiment_name,
        experiment_settings["predictions_file"]
    )
    with open(preditions_file, 'wb') as data_file:
        pickle.dump([y_true, y_pred], data_file)

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)


if __name__ == '__main__':
    main()
