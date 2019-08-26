"""
Runs a non-tracking experiment.

The arguments are loaded from a .yaml file, which is the input argument of this scirpt
(Instructions to run: `python run_non_tracking_experiment.py <path to .yaml file>`)
"""

import os
import sys
import logging
import yaml

from bff_positioning.data import Preprocessor, create_noisy_features, undersample_bf,\
    undersample_space, get_95th_percentile
from bff_positioning.models import CNN

# -------------------------------------------------------------------------------------------------
# Workaround TF logger problem in TF 1.14
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)  #pylint: disable=protected-access
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False               #pylint: disable=protected-access
except Exception:                                           #pylint: disable=broad-except
    pass
# Workaround TF logger problem in TF 1.14
# -------------------------------------------------------------------------------------------------


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
    model.set_graph()

    # Creates the validation set
    logging.info("Creating validation set...")
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
        features_train, labels_train = create_noisy_features(
            features,
            labels,
            experiment_settings,
            data_parameters,
        )
        model.train_epoch(features_train, labels_train)
        predictions_val = model.predict(features_val)
        keep_training, val_avg_dist = model.epoch_end(labels_val, predictions_val)
        # Upscales the validation score back to the original scale and gets the 95th percentile
        val_avg_dist *= data_parameters["pos_grid"][0]
        val_95_perc = get_95th_percentile(
            labels_val,
            predictions_val,
            rescale_factor=data_parameters["pos_grid"][0]
        )
        logging.info("Current avg val. distance: %.5f m || 95th percentile:  %.5f m\n",
            val_avg_dist, val_95_perc)

    # Store the trained model and cleans up
    logging.info("Saving and closing model.")
    experiment_name = os.path.basename(sys.argv[1]).split('.')[0]
    model.save(model_name=experiment_name)
    model.close()

if __name__ == '__main__':
    main()
