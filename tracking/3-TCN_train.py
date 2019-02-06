'''
TCN_train.py

-> Given the "paths" and the sorted dataset, trains a TCN. In the context of
    this script, the "validation set" uses the same paths as the training set,
    but with new beamformed fingerprints samples for each position. (i.e.
    measures the system's generalization capability for the trained paths)
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

from joblib import Parallel, delayed
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle


def check_valid_set_accuracy(predictor, X_valid, y_valid, batch_size):
    '''Returns the distances vector of the valid set'''
    distance_output = []

    n_batches_valid = math.ceil(len(X_valid) / batch_size)
    for batch_idx in range(n_batches_valid):

        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size
        #This system can only run on complete batches
        if end_idx > len(X_valid):
            break

        #Evaluates the distance
        distance_output.append(predictor.eval(feed_dict={
            X: X_valid[start_idx:end_idx],
            Y: y_valid[start_idx:end_idx],
            is_training: True   }))

    #flattens that list
    distance_output = [item for sublist in distance_output for item in sublist]

    return distance_output


def static_paths_sampler(mask, features, labels, time_steps, noise, scaler,
                         min_pow_cutoff):
    '''
    Processes a portion of the static paths
    '''
    X = []
    y = []

    possible_positions = len(labels)

    for idx in range(possible_positions):
        if not mask[idx]:
            continue

        #Applies the noise, the scaler, and the min_pow_cutoff. Appends the
        #   sequence to the (X, y) set if the sequence is valid.
        this_x = [features[idx, :]]*time_steps
        this_x = apply_noise_and_scaler(np.asarray(this_x), noise, scaler,
            min_pow_cutoff)
        if this_x is not None:
            y.append(labels[idx, :])
            X.append(this_x)

    return [X, y]



def moving_paths_sampler(mask, paths, features, labels, time_steps, noise,
                         scaler, min_pow_cutoff):
    '''
    Processes a portion of the non-static paths
    '''
    X = []
    y = []

    possible_positions = len(paths)

    for idx in range(possible_positions):
        if not mask[idx]:
            continue

        #Applies the noise, the scaler, and the min_pow_cutoff. Appends the
        #   sequence to the (X, y) set if the sequence is valid.
        this_sequence_of_indexes = paths[idx]
        this_x = []
        for index in this_sequence_of_indexes:
            this_x.append(features[index, :])
        assert len(this_x) == time_steps
        this_x = apply_noise_and_scaler(np.asarray(this_x), noise, scaler,
            min_pow_cutoff)
        if this_x is not None:
            y.append(labels[this_sequence_of_indexes[-1], :])
            X.append(this_x)

    return [X, y]


def sample_paths(paths, features, labels, time_steps, noise, min_pow_cutoff,
    scaler, sample_fraction=1.0, print_progress=False, parallel_jobs=1):
    '''
    Given the paths, the raw dataset, the time steps, the noise, the minimum
        detection power, and the scaler, returns:
        1 - X, the noisy tcn input data (sequences with time_steps length)
        2 - y, the predictions (one prediction per sequence)

    Having RAM problems? Use "sample_fraction" and train for more epochs :D
        The function will use ~sample_fraction times the original dataset
        per epoch, were 0 <= sample_fraction < 1
    '''

    X = []
    y = []

    assert sample_fraction <= 1.0 and sample_fraction >= 0.0, \
        'sample_fraction should be between 0.0 and 1.0!'

    #sample static paths
    if paths['s']:
        print(" (s) ", end = '', flush = True)
        possible_positions = len(labels)
        assert possible_positions == len(features)

        #creates a probabilistic mask (skip with probability 1-sample_fraction)
        mask = np.random.uniform(size=possible_positions)
        mask[mask < sample_fraction] = 1.0
        mask[mask < 1.0] = 0.0

        #gets the index separation
        position_split = np.linspace(0, possible_positions-1,
            num=parallel_jobs+1, dtype=int)

        #for each position, samples a sequence with size=time_steps. Uses multiple threads
        static_paths = Parallel(n_jobs=parallel_jobs, prefer='threads')\
            (delayed(static_paths_sampler)\
            (mask[position_split[idx]:position_split[idx+1]],
             features[position_split[idx]:position_split[idx+1], :],
             labels[position_split[idx]:position_split[idx+1], :],
             time_steps, noise, scaler, min_pow_cutoff) \
            for idx in range(len(position_split)-1))

        X_static, y_static = zip(*static_paths)
        for i in range(len(X_static)):
            X.extend(X_static[i])
            y.extend(y_static[i])
        del X_static, y_static, static_paths

    #sample pedestrian paths
    if paths['p']:
        print(" (p) ", end = '', flush = True)
        possible_positions = len(paths['p'])

        #creates a probabilistic mask (skip with probability 1-sample_fraction)
        mask = np.random.uniform(size=possible_positions)
        mask[mask < sample_fraction] = 1.0
        mask[mask < 1.0] = 0.0

        #gets the index separation
        position_split = np.linspace(0, possible_positions-1,
            num=parallel_jobs+1, dtype=int)

        #for each position, samples a sequence with size=time_steps. Uses multiple threads
        pedestrian_paths = Parallel(n_jobs=parallel_jobs, prefer='threads')\
            (delayed(moving_paths_sampler)\
            (mask[position_split[idx]:position_split[idx+1]],
             paths['p'][position_split[idx]:position_split[idx+1]],
             features, labels, time_steps, noise, scaler, min_pow_cutoff) \
            for idx in range(len(position_split)-1))

        X_ped, y_ped = zip(*pedestrian_paths)
        for i in range(len(X_ped)):
            X.extend(X_ped[i])
            y.extend(y_ped[i])
        del X_ped, y_ped, pedestrian_paths

    #sample car paths
    if paths['c']:
        print(" (c) ", end = '', flush = True)
        possible_positions = len(paths['c'])

        #creates a probabilistic mask (skip with probability 1-sample_fraction)
        mask = np.random.uniform(size=possible_positions)
        mask[mask < sample_fraction] = 1.0
        mask[mask < 1.0] = 0.0

        #gets the index separation
        position_split = np.linspace(0, possible_positions-1,
            num=parallel_jobs+1, dtype=int)

        #for each position, samples a sequence with size=time_steps. Uses multiple threads
        pedestrian_paths = Parallel(n_jobs=parallel_jobs, prefer='threads')\
            (delayed(moving_paths_sampler)\
            (mask[position_split[idx]:position_split[idx+1]],
             paths['c'][position_split[idx]:position_split[idx+1]],
             features, labels, time_steps, noise, scaler, min_pow_cutoff) \
            for idx in range(len(position_split)-1))

        X_ped, y_ped = zip(*pedestrian_paths)
        for i in range(len(X_ped)):
            X.extend(X_ped[i])
            y.extend(y_ped[i])
        del X_ped, y_ped, pedestrian_paths

    return X, y


def apply_noise_and_scaler(sequence, noise_std_converted, scaler, min_pow_cutoff):
    '''
    Given the sequence of features, the noise, the scaler, and the minimum
        detection power, creates a simulated sequence
    '''
    noise = np.random.normal(scale = noise_std_converted, size = sequence.shape)
    noisy_sequence = sequence + noise
    noisy_sequence[noisy_sequence < min_pow_cutoff] = 0

    #Checks if the sequence contains any empty sample. If it does,
    #   discards the sequence, as it is a broken sequence.
    this_sample_sums = np.sum(noisy_sequence, axis = 1)
    this_sample_sums[this_sample_sums > 0.001] = 1
    if np.sum(this_sample_sums) < len(sequence):
        return None

    #Applies the scaler
    noisy_sequence = scaler.fit_transform(noisy_sequence)
    if binary_scaler:   #(this variable comes from the outer scope)
        noisy_sequence = noisy_sequence.astype(bool)

    return noisy_sequence


if __name__ == "__main__":
    start = time.time()

    #Loads the TCN definition (which also loads the simulation parameters)
    print("\nInitializing the TCN graph...")
    exec(open("TCN_definition.py").read(), globals())

    #Loads the sorted dataset and the paths; Creates the scaler.
    print("\nLoading dataset and paths...")
    with open(preprocessed_file, 'rb') as f:
        features, labels, _ = pickle.load(f)
    with open(path_file, 'rb') as f:
        paths = pickle.load(f)

    if binary_scaler:
        scaler = Binarizer(0.1, copy=False)
    else:
        scaler = Normalizer(copy=False)

    #Now that the data is loaded and the TCN is defined, trains the TCN
    print("Starting the TF session... [Using GPU #{0}, sampling with {1} "\
        "thread(s)]".format(target_gpu, parallel_jobs))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        #Obtains the validation set
        print("Obtaining the validation set", end='', flush=True)
        X_valid, y_valid = sample_paths(paths, features, labels, time_steps,
            noise_std_converted, min_pow_cutoff, scaler, sample_fraction=0.1,
            print_progress=True, parallel_jobs=parallel_jobs)
        assert len(X_valid) == len(y_valid)
        X_valid, y_valid = shuffle(X_valid, y_valid)
        print(" Got {} validation sequences!".format(len(X_valid)))

        # Prepares the training phase
        current_batch = 0
        epochs_completed = 0
        sess.run(tf.global_variables_initializer())
        print("\nStarting the training!")
        print("All parameters:", int(np.sum([np.product([xi.value for xi in
            x.get_shape()]) for x in tf.global_variables()])))
        print("Trainable parameters:", int(np.sum([np.product([xi.value for xi in
            x.get_shape()]) for x in tf.trainable_variables()])))
        print_progress = True
        train_sample_args = [paths, features, labels, time_steps,
            noise_std_converted, min_pow_cutoff, scaler, 1.0/train_split,
            print_progress, parallel_jobs]

        # Main training loop
        while(epochs_completed < epochs):

            # Fetches the training set for the epoch
            print("Epoch {0}: Sampling paths".format(epochs_completed),
                end='', flush = True)
            X_train, y_train = sample_paths(*train_sample_args)
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
                    end_idx = len(X_train)

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

            # Assesses the validation performance
            # "distance" -> the predictor
            distance_output = check_valid_set_accuracy(distance, X_valid,
                y_valid, batch_size)
            avg_distance = np.mean(distance_output)
            sorted_distance = np.sort(distance_output)
            distance_95 = sorted_distance[int(len(distance_output) * 0.95)]

            print('; Finished Epoch {0},   valid distance (m)= {1:.4f},' \
                '   95% percentile (m) = {2:.4f},   next LR = {3:.4e}' \
                .format(epochs_completed-1, avg_distance, distance_95,
                learning_rate))

        #---------------------------------------------------------------------- [end of training]
        #----------------------------------------------------------------------

        #Saves the model
        session_name = 'tcn_noise_' + str(test_noise) + '_length_' + str(time_steps)
        if not os.path.exists('results/'):
            os.makedirs('results/')
        saver.save(sess, 'results/' + session_name)

        #after last epoch -> run for a complete validation set
        print("Final check - running for the whole validation set!")
        X_valid, y_valid = sample_paths(paths, features, labels, time_steps,
            noise_std_converted, min_pow_cutoff, scaler,)
        distance_output = check_valid_set_accuracy(distance, X_valid,
                y_valid, batch_size)
        avg_distance = np.mean(distance_output)
        sorted_distance = np.sort(distance_output)
        distance_95 = sorted_distance[int(len(distance_output) * 0.95)]

        print('Validation distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}' \
            .format(avg_distance, distance_95))


    #After training the TCN, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))