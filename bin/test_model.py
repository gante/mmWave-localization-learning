"""
Runs the test dataset over the data, and stores the predictions

The arguments are loaded from a .yaml file, which is the input argument of this script
(Instructions to run: `python test_model.py <path to .yaml file>`)
"""

import os
import sys
import time
import logging
import pickle
import yaml

import numpy as np
from tqdm import tqdm

from bff_positioning.data import Preprocessor, PathCreator, create_noisy_features, \
    get_95th_percentile, undersample_bf, undersample_space, sample_paths
from bff_positioning.models import CNN, LSTM, TCN
from bff_positioning.models.metrics import score_predictions


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
    path_parameters = experiment_config['path_parameters'] \
        if 'path_parameters' in experiment_config else None

    # Loads the raw dataset
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()
    if path_parameters:
        path_creator = PathCreator(data_parameters, path_parameters, labels)
        paths = path_creator.load_paths()

    # Undersamples the dataset (if requested)
    if "undersample_bf" in experiment_settings and experiment_settings["undersample_bf"]:
        features = undersample_bf(features, data_parameters["beamformings"])
    if "undersample_space" in experiment_settings:
        assert not path_parameters, "This option is not supported for tracking experiments, "\
            "unless the code for the path creation is updated"
        features, labels = undersample_space(features, labels, data_parameters["undersample_space"])

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

    # Prediction loop
    mc_dropout_samples = ml_parameters.get("mc_dropout", 0)
    if mc_dropout_samples:
        logging.info("Evaluation mode: MC Dropout sampling")
        tests_per_input = 1
    elif "tests_per_position" in experiment_settings:
        logging.info("Evaluation mode: Single-point position estimates")
        tests_per_input = experiment_settings["tests_per_position"]
    else:
        logging.info("Evaluation mode: Path-based position estimates")
        logging.info("Note - each set of paths will be split into 10 sub-sets, for easier RAM"
            "management -- that's why you'll see 10x test sets in the next logging messages.")
        tests_per_input = experiment_settings["tests_per_path"] * 10

    y_true = []
    y_pred = []
    for set_idx in range(tests_per_input):
        logging.info("Creating test set %2s out of %2s...", set_idx+1, tests_per_input)
        if path_parameters:
            features_test, labels_test, _ = sample_paths(
                paths["test"],
                features,
                labels,
                experiment_settings,
                data_parameters,
                path_parameters,
                sample_fraction=0.1
            )
        else:
            features_test, labels_test = create_noisy_features(
                features,
                labels,
                experiment_settings,
                data_parameters,
            )
        logging.info("Running predictions and storing data...\n")

        y_true.append(labels_test)
        if not mc_dropout_samples: # MC Dropout OFF
            predictions_test = model.predict(features_test)
            y_pred.append(predictions_test)
        else: # MC Dropout ON
            for sample_rnd in tqdm(range(mc_dropout_samples)):
                predictions_test = model.predict(features_test)
                y_pred.append(predictions_test)

    # Stack results and sanity check
    y_true = np.vstack(y_true)
    if mc_dropout_samples:
        y_pred = np.stack(y_pred, axis=2)
    else:
        y_pred = np.vstack(y_pred)
    assert y_true.shape[0] == y_pred.shape[0], \
        "The predictions and the labels must have the same number of examples!"
    assert y_true.shape[1] == y_pred.shape[1], \
        "The number of dimensions per sample must stay constant!"

    # Closes the model, gets the test scores, and stores predictions-labels pairs
    model.close()
    if not mc_dropout_samples: # Doesn't make sense to test MCDropout samples, they underperform :)
        logging.info("Computing test metrics...")
        test_score = score_predictions(y_true, y_pred, ml_parameters["validation_metric"])
        test_score *= data_parameters["pos_grid"][0]
        test_95_perc = get_95th_percentile(
            y_true,
            y_pred,
            rescale_factor=data_parameters["pos_grid"][0]
        )
        logging.info("Average test distance: %.5f m || 95th percentile: %.5f m\n",
            test_score, test_95_perc)

    preditions_file = os.path.join(
        ml_parameters["model_folder"],
        experiment_name + '_' + experiment_settings["predictions_file"]
    )
    with open(preditions_file, 'wb') as data_file:
        pickle.dump([y_true, y_pred], data_file)

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)


if __name__ == '__main__':
    main()
