"""
Trains a model, acording to the experiment file input

The arguments are loaded from a .yaml file, which is the input argument of this script
(Instructions to run: `python train_model.py <path to .yaml file>`)
"""

import os
import sys
import time
import logging
import yaml

from bff_positioning.data import Preprocessor, PathCreator, create_noisy_features, undersample_bf,\
    undersample_space, get_95th_percentile, sample_paths
from bff_positioning.models import CNN, LSTM, TCN


def main():
    """Main block of code, which runs the training"""

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

    # Initializes the model and prepares it for training
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
    model.set_graph()

    # Creates the validation set
    logging.info("Creating validation set...")
    if path_parameters:
        features_val, labels_val, _ = sample_paths(
            paths["validation"],
            features,
            labels,
            experiment_settings,
            data_parameters,
            path_parameters,
        )
    else:
        features_val, labels_val = create_noisy_features(
            features,
            labels,
            experiment_settings,
            data_parameters,
        )

    # Runs the training loop
    logging.info("\nStaring the training loop!\n")
    keep_training = True
    while keep_training:
        logging.info("Creating noisy set for this epoch...")
        if path_parameters:
            features_train, labels_train, _ = sample_paths(
                paths["train"],
                features,
                labels,
                experiment_settings,
                data_parameters,
                path_parameters,
                sample_fraction=experiment_settings["train_sample_fraction"]
            )
        else:
            features_train, labels_train = create_noisy_features(
                features,
                labels,
                experiment_settings,
                data_parameters,
            )
        model.train_epoch(features_train, labels_train)
        predictions_val = model.predict(features_val, validation=True)
        keep_training, val_avg_dist = model.epoch_end(labels_val, predictions_val)
        if predictions_val is not None:
            # Upscales the validation score back to the original scale and gets the 95th percentile
            val_avg_dist *= data_parameters["pos_grid"][0]
            val_95_perc = get_95th_percentile(
                labels_val,
                predictions_val,
                rescale_factor=data_parameters["pos_grid"][0]
            )
            logging.info("Current avg val. distance: %.5f m || 95th percentile: %.5f m\n",
                val_avg_dist, val_95_perc)

    # Store the trained model and cleans up
    logging.info("Saving and closing model.")
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    model.save(model_name=experiment_name)
    model.close()

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)

if __name__ == '__main__':
    main()
