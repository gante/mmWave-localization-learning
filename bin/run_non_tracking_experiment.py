"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python run_non_tracking_experiment.py <path to .yaml file>`)
"""

import sys
import logging
import yaml

from bff_positioning.data import Preprocessor, create_noisy_features, undersample_bf,\
    undersample_space
from bff_positioning.models import CNN

def main():
    """Main block of code, which runs the experiment"""

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

    # Loads the dataset
    logging.info("Loading the dataset...")
    data_preprocessor = Preprocessor(data_parameters)
    features, labels = data_preprocessor.load_dataset()

    # Undersamples the dataset (if requested)
    if "undersample_bf" in experiment_settings and experiment_settings["undersample_bf"]:
        features = undersample_bf(features, data_parameters["beamformings"])
    if "undersample_space" in experiment_settings:
        features, labels = undersample_space(features, labels, data_parameters["undersample_space"])

    # Initializes the model and prepares it for training
    logging.info("Initializing the model...")
    if experiment_settings["model_type"] == "cnn":
        model = CNN(ml_parameters)
    else:
        raise ValueError("The simulation settings specified 'model_type'={}. Currently, only "
            "'cnn' is supported. [If you were looking for the HCNNs: sorry, the code was quite "
            "lengthy, so I moved its refactoring into a future to do. Please contact me if you "
            "want to experiment with it.]".format(experiment_settings["model_type"]))
    model.set_graph(features.shape(), labels.shape())

    # Creates the validation set
    logging.info("Creating validation set...")
    features_val, labels_val = create_noisy_features(
        features,
        labels,
        data_parameters,
        experiment_settings
    )

    # Runs the training loop
    logging.info("\nStaring the training loop!\n")
    keep_training = True
    while keep_training:
        logging.info("\nCreating noisy set for this epoch...")
        features_train, labels_train = create_noisy_features(
            features,
            labels,
            data_parameters,
            experiment_settings
        )
        model.train_epoch(features_train, labels_train)
        keep_training, val_score = model.epoch_end(features_val, labels_val)
        logging.info("Current average validation distance: %s meters", val_score)

    # Store the trained model and cleans up
    logging.info("Saving and closing model.")
    experiment_name = experiment_settings["model_type"] + '_' + experiment_settings["noise_std"]
    model.save(model_name=experiment_name)
    model.close()

if __name__ == '__main__':
    main()
