'''
TCN_train.py

-> Given the "paths" and the sorted dataset, trains a model. In the context of
    this script, the "validation set" uses the same paths as the training set,
    but with new beamformed fingerprints samples for each position. (i.e.
    measures the system's generalization capability for the seen paths)
'''
######### Suppresses TF warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import os
import time
import pickle
import math

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import Binarizer
from sklearn.utils import shuffle

from path_sampling_functions import sample_paths, check_accuracy


if __name__ == "__main__":
    start = time.time()

    #Loads the model definition (which also loads the simulation parameters)
    print("\nInitializing the model graph...")
    exec(open("model_definition.py").read(), globals())
    tf_dict = {'distance': distance,
               'X': X,
               'Y': Y,
               'is_training': is_training}
    if use_tcn:
        print("A TCN model was initialized!")
    else:
        print("A LSTM model was initialized!")

    #Loads the sorted dataset and the paths; Creates the scaler.
    print("\nLoading dataset and paths...")
    with open(preprocessed_file, 'rb') as f:
        features, labels, _ = pickle.load(f)
    with open(path_file_train, 'rb') as f:
        paths_train = pickle.load(f)
    with open(path_file_valid, 'rb') as f:
        paths_valid = pickle.load(f)

    if binary_scaler:
        scaler = Binarizer(0.1, copy=False)
    else:
        raise NotImplementedError

    #Now that the data is loaded and the model is defined, trains the model
    print("Starting the TF session... [Using GPU #{0}, sampling with {1} "\
        "thread(s)]".format(target_gpu, parallel_jobs))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        print("All parameters:", int(np.sum([np.product([xi.value for xi in
            x.get_shape()]) for x in tf.global_variables()])))
        print("Trainable parameters:", int(np.sum([np.product([xi.value for xi in
            x.get_shape()]) for x in tf.trainable_variables()])))
        print_progress = True

        #Obtains the validation set
        print("Obtaining the validation set", end='', flush=True)
        X_valid, y_valid, _ = sample_paths(paths_valid, features, labels,
            time_steps, noise_std_converted, min_pow_cutoff, scaler,
            print_progress=True, parallel_jobs=parallel_jobs)
        assert len(X_valid) == len(y_valid)
        X_valid, y_valid = shuffle(X_valid, y_valid)
        del paths_valid
        print(" Got {} validation sequences!".format(len(X_valid)))

        # Prepares the training phase
        epochs_completed = 0
        keep_training = True
        epochs_not_improving = 0
        best_valid_distance = 999999.9
        best_95_distance = 999999.9
        sess.run(tf.global_variables_initializer())

        model_type = 'tcn' if use_tcn else 'lstm'
        session_name = model_type + '_noise_' + str(int(test_noise)) + \
            '_length_' + str(time_steps)
        if not os.path.exists('results/'):
            os.makedirs('results/')

        print("\nStarting the training!")
        train_sample_args = [paths_train, features, labels, time_steps,
            noise_std_converted, min_pow_cutoff, scaler, 1.0/train_split,
            print_progress, parallel_jobs]

        # Main training loop
        while(keep_training):

            # Fetches the training set for the epoch
            print("Epoch {0}: Sampling paths".format(epochs_completed),
                end='', flush = True)
            X_train, y_train, _ = sample_paths(*train_sample_args)
            assert len(X_train) == len(y_train)
            X_train, y_train = shuffle(X_train, y_train)
            n_batches_train = math.ceil(len(X_train) / batch_size)

            print(" Training", end='', flush = True)
            progress = 0.0
            for batch_idx in range(n_batches_train):

                #prints a "." every 10% of an epoch
                if (batch_idx / n_batches_train) >= progress + 0.1:
                    print(".", end = '', flush = True)
                    progress += 0.1

                start_idx = batch_idx * batch_size
                end_idx = (batch_idx+1) * batch_size
                if end_idx > len(X_train):
                    #for LSTMs, the last batch can't be executed if the batch
                    #   size differs
                    if use_tcn:
                        end_idx = len(X_train)
                    else:
                        continue

                #Expected input shape for the input:
                #   (shape=[batch_size/None, time_steps, single_element_shape])
                train_step.run(feed_dict={X: X_train[start_idx:end_idx],
                                          Y: y_train[start_idx:end_idx],
                                          learning_rate_var: learning_rate,
                                          is_training: True})

            # Training "epoch" finished
            del X_train, y_train
            epochs_completed += 1
            learning_rate = learning_rate * learning_rate_decay

            if epochs_completed % valid_assessment_period == 0:
                # Assesses the validation performance
                distance_output = check_accuracy(X_valid, y_valid, batch_size, tf_dict)
                avg_distance = np.mean(distance_output)
                sorted_distance = np.sort(distance_output)
                distance_95 = sorted_distance[int(len(distance_output) * 0.95)]

                print(';   Validation distance (m) = {1:.4f},' \
                    '   95% percentile (m) = {2:.4f},   next LR = {3:.4e}' \
                    .format(epochs_completed-1, avg_distance, distance_95,
                    learning_rate))

                #Stores the model if it beats the previous best model by more
                #   than 1%. Otherwise, evaluates the early stopping mechanism
                if avg_distance < (best_valid_distance * 0.99):
                    best_valid_distance = avg_distance
                    best_95_distance = distance_95
                    epochs_not_improving = 0
                    #saves the model
                    saver.save(sess, 'results/' + session_name)
                elif epochs_completed > epochs_hard_cap:
                    keep_training = False
                else:
                    epochs_not_improving += 1
                    if epochs_not_improving >= early_stopping:
                        keep_training = False
            else:
                print("")

        #---------------------------------------------------------------------- [end of training]
        #----------------------------------------------------------------------

    #After training the model, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("\nEnd of the training, printing stored results.")
    print("Validation distance (m) = {0:.4f},   95% percentile (m) = {1:.4f}"\
        .format(best_valid_distance, best_95_distance))
    print("Execution time = {0:.4}s".format(exec_time))