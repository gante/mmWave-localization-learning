######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import tensorflow as tf
import pickle
import numpy as np


n_predictions = 10





#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("../simulation_parameters.py").read(), globals())
data_file =  '../' + data_file

#Runs "load_data.py" and keeps its variables                [data loading]
exec(open("../load_data.py").read(), globals())



if binary_scaler == False: print("WARNING - the non-zeros vector will contain incorrect values, since the used scaler isnt binary!")


with tf.Session() as sess:
    
    # defines the saver (aka loader)
    session_name = 'nn_result_' + scaler_name + '_%.2f' % test_noise
    saver = tf.train.import_meta_graph('../results/' + session_name + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../results./'))
    
    # Redefines the needed stuff [i.e. placeholders AND operations, that must be saved by name]
    graph = tf.get_default_graph()
    
    input = graph.get_tensor_by_name("input:0")
    position = graph.get_tensor_by_name("position:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    
    distance = graph.get_tensor_by_name("distance:0")
    
    #now that the session is loaded, predicts stuff!
    
    
    #evaluates the NN on the test set [since the CNN takes a lot of resources, a loop must be used!]
    print("Predicting the results", end='', flush = True)
    
    distance_output = []
    non_zeros = []
    for i in range(n_predictions):
    
        print(".", end='', flush = True)
        features_test, labels_test = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler)
        
        end_test = 0
        j = 0
        while(end_test < features_test.shape[0]):
            start_test = j*test_batch_size
            end_test = (j+1) *test_batch_size
            j = j+1
            
            if(end_test > features_test.shape[0]):
                end_test = features_test.shape[0]
            
            distance_output.append(distance.eval(feed_dict={
                input: features_test[start_test:end_test], position: labels_test[start_test:end_test], keep_prob: 1.0}))
                # input: features_test[start_test:end_test], position: labels[start_test:end_test], keep_prob: 1.0})) # <--- WRONG
            
            #stores the corresponding labels
            if (i == 0) and (j==1):
                all_labels = labels_test[start_test:end_test]
            else:
                all_labels = np.append(all_labels, labels_test[start_test:end_test], axis = 0)
                
                
            #stores the corresponding number of non-zeros [this operation assumes binary scaler]
            for k in range(start_test, end_test): 
                non_zeros.append(np.sum(features_test[k]))
            
    #flattens that list
    distance_output = [item for sublist in distance_output for item in sublist]
    avg_distance = np.mean(distance_output)
    sorted_distance = np.sort(distance_output)
    distance_95 = sorted_distance[int(features_test.shape[0] * 0.95 * n_predictions)]
    print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))
    
    
#Storing the result
#[but before storing, double checks]

distance_output = np.asarray(distance_output)
non_zeros = np.asarray(non_zeros)

assert all_labels.shape[0] == distance_output.shape[0]
assert all_labels.shape[0] == non_zeros.shape[0]

print("Storing the result ...")
with open('predictions.pkl', 'wb') as f:
    pickle.dump([distance_output, all_labels, non_zeros], f)
    