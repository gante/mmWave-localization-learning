'''
path_sampling_functions.py

-> Holds functions to generate sequences from the paths configuration and the
    ray-tracing data
'''

import numpy as np
from joblib import Parallel, delayed
import math

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
        3 - a dict with the ending index for that path type

    Having RAM problems? Use "sample_fraction" and train for more epochs :D
        The function will use ~sample_fraction times the original dataset
        per epoch, were 0 <= sample_fraction < 1
    '''

    X = []
    y = []
    paths_type_delimiter = {'s':0, 'p':0, 'c':0}

    assert sample_fraction <= 1.0 and sample_fraction >= 0.0, \
        'sample_fraction should be between 0.0 and 1.0!'

    #sample static paths
    if paths['s']:
        if print_progress:
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
        paths_type_delimiter['s'] = len(X)
        paths_type_delimiter['p'] = len(X)
        paths_type_delimiter['c'] = len(X)
        del X_static, y_static, static_paths

    #sample pedestrian paths
    if paths['p']:
        if print_progress:
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
        paths_type_delimiter['p'] = len(X)
        paths_type_delimiter['c'] = len(X)
        del X_ped, y_ped, pedestrian_paths

    #sample car paths
    if paths['c']:
        if print_progress:
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
        car_paths = Parallel(n_jobs=parallel_jobs, prefer='threads')\
            (delayed(moving_paths_sampler)\
            (mask[position_split[idx]:position_split[idx+1]],
             paths['c'][position_split[idx]:position_split[idx+1]],
             features, labels, time_steps, noise, scaler, min_pow_cutoff) \
            for idx in range(len(position_split)-1))

        X_car, y_car = zip(*car_paths)
        for i in range(len(X_car)):
            X.extend(X_car[i])
            y.extend(y_car[i])
        paths_type_delimiter['c'] = len(X)
        del X_car, y_car, car_paths

    return X, y, paths_type_delimiter


def apply_noise_and_scaler(sequence, noise_std_converted, scaler,
                           min_pow_cutoff, binary_scaler=True):
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
    if binary_scaler:
        noisy_sequence = noisy_sequence.astype(bool)

    return noisy_sequence


def check_accuracy(X_valid, y_valid, batch_size, tf_dict, use_tcn):
    '''Returns the distances vector of a given set'''

    #Expands the TF_dict (containg graph references)
    distance = tf_dict['distance']
    X = tf_dict['X']
    Y = tf_dict['Y']
    is_training = tf_dict['is_training']

    distance_output = []

    n_batches_valid = math.ceil(len(X_valid) / batch_size)
    for batch_idx in range(n_batches_valid):

        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        if end_idx > len(X_valid):
            if not use_tcn:
                #This system can only run on complete batches
                break
            else:
                end_idx = len(X_valid)

        #Evaluates the distance
        distance_output.append(distance.eval(feed_dict={
            X: X_valid[start_idx:end_idx],
            Y: y_valid[start_idx:end_idx],
            is_training: True   }))

    #flattens that list
    distance_output = [item for sublist in distance_output for item in sublist]

    return distance_output