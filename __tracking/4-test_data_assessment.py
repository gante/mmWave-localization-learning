'''
test_data_assessment.py

-> Uses the trained sequence-based model on a new, unseen set of paths.
    It allows us to see the complete generalization capabilities of the model.
'''
######### Suppresses TF warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########

import tensorflow as tf
import pickle
import numpy as np
import time

from sklearn.preprocessing import Binarizer

from path_sampling_functions import sample_paths, check_accuracy


def print_partial_stats(path_type, distance_vector):
    '''Prints the partial stats for a given path type'''
    avg_distance = np.mean(distance_vector)
    sorted_distance = np.sort(distance_vector)
    assessed_paths = len(distance_vector)
    distance_95 = sorted_distance[int(assessed_paths * 0.95)]
    print(path_type + ': average distance (m) = {0:.4f},    '\
        '95% percentile (m) = {1:.4f},    assesed sequences: {2}' \
        .format(avg_distance, distance_95, assessed_paths))


if __name__ == "__main__":
    start = time.time()

    assert os.path.exists('results/static_results.pkl') == False, \
        "There cannot be previous results (this script appends to files)"
    assert os.path.exists('results/ped_results.pkl') == False, \
        "There cannot be previous results (this script appends to files)"
    assert os.path.exists('results/car_results.pkl') == False, \
        "There cannot be previous results (this script appends to files)"

    #Runs "simulation_parameters.py" and keeps its variables
    exec(open("simulation_parameters.py").read(), globals())
    batch_size = tcn_parameters['batch_size'] if use_tcn else \
        lstm_parameters['batch_size']

    #Loads the sorted dataset and the paths; Creates the scaler.
    print("\nLoading dataset and paths...")
    with open(preprocessed_file, 'rb') as f:
        features, labels, _ = pickle.load(f)
    with open(path_file_test, 'rb') as f:
        paths = pickle.load(f)

    if binary_scaler:
        scaler = Binarizer(0.1, copy=False)
    else:
        raise NotImplementedError

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        # defines the loader ("saver" in TF language)
        model_type = 'tcn' if use_tcn else 'lstm'
        session_name = model_type + '_noise_' + str(int(test_noise)) + \
            '_length_' + str(time_steps)
        saver = tf.train.import_meta_graph('results/' + session_name + '.meta')
        saver.restore(sess, 'results/' + session_name)
        print("Loading model with name = '{}'".format(session_name))

        # Redefines the needed stuff
        # [i.e. placeholders AND operations, that were saved by name]
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        distance = graph.get_tensor_by_name("distance:0")
        tf_dict = {'distance': distance,
                   'X': X,
                   'Y': Y,
                   'is_training': is_training}

        #Now that the session is loaded, predicts stuff! Will test each path
        #   'n_test' times on average. However, to avoid consuming too much RAM,
        #   each round is split in 'test_split' parts.
        prediction_rounds = n_tests * test_split

        distance_output_static = []
        distance_output_ped = []
        distance_output_car = []

        y_output_static = []
        y_output_ped = []
        y_output_car = []

        non_zero_static = []
        non_zero_ped = []
        non_zero_car = []

        for i in range(prediction_rounds):
            print("\nPrediction round #{0} out of {1}...".format(i+1, prediction_rounds))
            X_test, y_test, paths_delimiter = sample_paths(paths, features,
                labels, time_steps, noise_std_converted, min_pow_cutoff,
                scaler, sample_fraction=1.0/test_split,
                print_progress=False, parallel_jobs=parallel_jobs)

            #Static paths
            if paths_delimiter['s'] > 0:
                this_distance_static = check_accuracy(
                    X_test[0:paths_delimiter['s']],
                    y_test[0:paths_delimiter['s']],
                    batch_size, tf_dict, use_tcn)
                distance_output_static += this_distance_static
                print_partial_stats('Static paths    ', distance_output_static)
                y_output_static = y_test[0:paths_delimiter['s']]
                non_zero_static = np.sum(X_test[0:paths_delimiter['s']], axis=(1,2))
                assert len(non_zero_static) == len(y_output_static)
                assert len(this_distance_static) == len(non_zero_static)

            #Pedestrian paths
            if paths_delimiter['p'] > paths_delimiter['s']:
                this_distance_ped = check_accuracy(
                    X_test[paths_delimiter['s']:paths_delimiter['p']],
                    y_test[paths_delimiter['s']:paths_delimiter['p']],
                    batch_size, tf_dict, use_tcn)
                distance_output_ped += this_distance_ped
                print_partial_stats('Pedestrian paths', distance_output_ped)
                y_output_ped = y_test[paths_delimiter['s']:paths_delimiter['p']]
                non_zero_ped = np.sum(X_test[paths_delimiter['s']:paths_delimiter['p']], axis=(1,2))
                assert len(non_zero_ped) == len(y_output_ped)
                assert len(this_distance_ped) == len(non_zero_ped)

            #Car paths
            if paths_delimiter['c'] > paths_delimiter['p']:
                this_distance_car = check_accuracy(
                    X_test[paths_delimiter['p']:paths_delimiter['c']],
                    y_test[paths_delimiter['p']:paths_delimiter['c']],
                    batch_size, tf_dict, use_tcn)
                distance_output_car += this_distance_car
                print_partial_stats('Car paths       ', distance_output_car)
                y_output_car = y_test[paths_delimiter['p']:paths_delimiter['c']]
                non_zero_car = np.sum(X_test[paths_delimiter['p']:paths_delimiter['c']], axis=(1,2))
                assert len(non_zero_car) == len(y_output_car)
                assert len(this_distance_car) == len(non_zero_car)

            print_partial_stats('All paths       ', distance_output_car + \
                distance_output_ped + distance_output_static)

            #After each prediction, stores the results
            print("Storing results...")
            with open('results/static_results.pkl', 'ab') as f:
                pickle.dump([this_distance_static, y_output_static, non_zero_static], f)
            with open('results/ped_results.pkl', 'ab') as f:
                pickle.dump([this_distance_ped, y_output_ped, non_zero_ped], f)
            with open('results/car_results.pkl', 'ab') as f:
                pickle.dump([this_distance_car, y_output_car, non_zero_car], f)

    #After all predictions, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))
