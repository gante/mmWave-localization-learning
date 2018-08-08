######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import time
import numpy as np
import pickle
import math
import tensorflow as tf
import argparse
import os
from sklearn.utils import shuffle



def train_a_class(index, print_periodicity, print_train):

    #Runs "DNN_definition.py" and keeps its variables           [Loads static NN stuff]
    exec(open("DNN_definition_regr.py").read(), globals())
    # epochs = 1 -> quick dbg
    
    #The program crashes without the following line (and it shouldn't)
    lr = dictionary['learning_rate']

    print("\n\nTRAINING FOR CLASS #{0}".format(index))


    ##########################################################
    #   Overwrites old variables
    #class -> class central position
    #index = (y_index * lateral_partition) + x_index
    x_index = index % lateral_partition
    y_index = (index - x_index) / lateral_partition
    x_pos = ( (x_index + 0.5) /lateral_partition)
    y_pos = ( (y_index + 0.5) /lateral_partition)
    # print(x_pos, y_pos)
    
    #Overwrites the NN's last layer based on the class index
    W_fc_l = weight_variable([fcl_neurons, 2], std = 0.01)
    bias_x = bias_variable([1], val = x_pos)
    bias_y = bias_variable([1], val = y_pos)
    b_fc_l = tf.concat([bias_x, bias_y],0)
    prediction_l = tf.matmul(h_fc_drop[-1], W_fc_l) + b_fc_l          # learns with the original value, gets the final results with the clipped version
    prediction_clip = tf.clip_by_value(prediction_l, 0.0, 1.0)
    
    
    # TF functions
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                         # Add for batch Normalization
    with tf.control_dependencies(update_ops):                                       # Add for batch Normalization
    
        delta_positions = location - prediction_clip 		            # contains delta_x AND delta_y
        delta_positions_squared = tf.square(delta_positions * data_downscale) 	    
                                                                        # contains delta_x^2 AND delta_y^2, SCALING UP BACK TO THE ORIGINAL RANGE [0;data_downscale]
        distance_square = tf.reduce_sum(delta_positions_squared,1)      # reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
        distance = tf.sqrt(distance_square, name='distance')	        # and then computes its square root

        # loss as MMSE
        loss_function = tf.reduce_mean(tf.square(location - prediction_l))
            
        # defines the optimizer (ADAM)
        train_step_l = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss_function)
    
    
    ##########################################################
    #   Loads the data
    file_name = 'class_data/class_' + str(index)

    features = []
    labels = []
    softmax = []
    min_dataset = 9999999

    with open(file_name, 'rb') as f:
        for i in range(n_predictions):
            this_features, this_labels, this_softmax = pickle.load(f)
            features.append(this_features)
            labels.append(this_labels)
            softmax.append(this_softmax)
            if this_features.shape[0] < min_dataset:
                min_dataset = this_features.shape[0]


    if min_dataset > 0:
        
        ##########################################################
        #   Runs the training

        #Starts the Session
        saver = tf.train.Saver()
        with tf.Session() as sess:
          
          
            #initializes the variables
            print("Initializing the variables and starting the training...")
            sess.run(tf.global_variables_initializer())
          
          
            #defines the features/labels
            features_train = features[0]
            if binary_scaler:
                features_train = np.asarray(features_train, dtype = np.float32)
            labels_train = labels[0]
            softmax_train = softmax[0]
            features_train, labels_train, softmax_train = shuffle(features_train, labels_train, softmax_train)
            train_set_size = features_train.shape[0]
            num_batches = int(math.ceil(train_set_size / batch_size))
            assert features_train.shape[0] == labels_train.shape[0]
            assert softmax_train.shape[0] == features_train.shape[0]
          
            #runs the training phase
            current_batch = 0
            epochs_completed = 0
            epochs_not_improving = 0
            best_acc = 9999999.0
            epochs_not_improving_limit = math.ceil(50/print_periodicity) + 1 #min = 2 checks
                
            while((epochs_completed < epochs) and (epochs_not_improving < epochs_not_improving_limit)):
            
                start_batch = current_batch*batch_size
                if ((current_batch+1)*batch_size) > train_set_size:
                    batch = [features_train[start_batch:], 
                            labels_train[start_batch:],
                            softmax_train[start_batch:]]
                else:
                    end_batch = ((current_batch+1)*batch_size)
                    batch = [features_train[start_batch:end_batch], 
                            labels_train[start_batch:end_batch],
                            softmax_train[start_batch:end_batch]]
                
                  
                #trains the network (with dropout)
                train_step_l.run(feed_dict={input: batch[0], location: batch[1], softmax_predictions: batch[2],
                                    keep_prob: 1.0-dropout, learning_rate_var: lr, phase: 1})
                
                
                #If the epoch is finished
                current_batch += 1
                if(current_batch >= num_batches):
                                    
                    current_batch = 0
                    epochs_completed += 1
                    
                    #Prints a life signal each epoch
                    if epochs_completed % 10 == 0:
                        print('|', end = '', flush = True)
                    else:
                        print('.', end = '', flush = True)
                    
                    #Prints the test set results every x epochs
                    if (epochs_completed % print_periodicity == 0) or (epochs_completed == epochs):
                        #evaluates the NN on the test set [since the CNN takes a lot of resources, a loop mas be used!]
                        distance_output = []
                        non_zeros = []
                        
                        #repeats the following for each test set
                        for k in range(test_sets):
                            
                            #loads the test set
                            features_test = features[-(k+1)]
                            labels_test = labels[-(k+1)]
                            softmax_test = softmax[-(k+1)]
                            
                            if binary_scaler:
                                features_test = np.asarray(features_test, dtype = np.float32)
                            
                            end_test = 0
                            j = 0
                            while(end_test < features_test.shape[0]):
                                start_test = j*test_batch_size
                                end_test = (j+1) *test_batch_size
                                j = j+1
                                
                                if(end_test > features_test.shape[0]):
                                    end_test = features_test.shape[0]
                                
                                distance_output.append(distance.eval(feed_dict={input: features_test[start_test:end_test], 
                                    location: labels_test[start_test:end_test], softmax_predictions: softmax_test[start_test:end_test], 
                                    keep_prob: 1.0, phase: 0}))
                                    
                                #stores the corresponding number of non-zeros [this operation assumes binary scaler]
                                for k in range(start_test, end_test): 
                                    non_zeros.append(np.sum(features_test[k]))
                        
                        #flattens that list
                        distance_output = [item for sublist in distance_output for item in sublist]
                        avg_distance = np.mean(distance_output)
                        sorted_distance = np.sort(distance_output)
                        distance_95 = sorted_distance[int(sorted_distance.shape[0] * 0.95)]
                        
                        if print_train:
                            #repeats the same, but for train data
                            distance_train = []
                            end_test = 0
                            j = 0
                            while(end_test < features_train.shape[0]):
                                start_test = j*test_batch_size
                                end_test = (j+1) *test_batch_size
                                j = j+1
                                
                                if(end_test > features_train.shape[0]):
                                    end_test = features_train.shape[0]
                                
                                distance_train.append(distance.eval(feed_dict={input: features_train[start_test:end_test], 
                                    location: labels_train[start_test:end_test], softmax_predictions: softmax_train[start_test:end_test], 
                                    keep_prob: 1.0, phase: 0}))
                                    
                            #flattens that list
                            distance_train = [item for sublist in distance_train for item in sublist]
                            avg_train = np.mean(distance_train)
                            
                            print('Finished Epoch {0},   Test/Train distance (m)= {1:.4f}/{4:.4f},   95% percentile (m) = {2:.4f},   next LR = {3:.4e}'
                                .format(epochs_completed, avg_distance, distance_95, lr, avg_train))
                        else:
                            print('Finished Epoch {0},   Test distance (m)= {1:.4f},   95% percentile (m) = {2:.4f},   next LR = {3:.4e}'
                                .format(epochs_completed, avg_distance, distance_95, lr))
                                
                        if avg_distance < best_acc:
                            epochs_not_improving = 0
                            best_acc = avg_distance
                        else:
                            epochs_not_improving += 1
                            
                        
                    
                    #loads a new train set
                    set_to_load = epochs_completed % train_sets
                    features_train = features[set_to_load]
                    if binary_scaler:
                        features_train = np.asarray(features_train, dtype = np.float32)
                    labels_train = labels[set_to_load]
                    softmax_train = softmax[set_to_load]
                    features_train, labels_train, softmax_train = shuffle(features_train, labels_train, softmax_train)
                    train_set_size = features_train.shape[0]
                    num_batches = int(math.ceil(train_set_size / batch_size))
                    assert features_train.shape[0] == labels_train.shape[0]
                    assert softmax_train.shape[0] == features_train.shape[0]
                    
                    #Adapt the learning rate
                    lr = lr * learning_rate_decay
            
            
            #Prints the final statistics for the test sets:
            correct_class = []
            correct_distance = []
            incorrect_distance = []
            
            for k in range(test_sets):
                            
                #loads the test set
                features_test = features[-(k+1)]
                labels_test = labels[-(k+1)]
                softmax_test = softmax[-(k+1)]
                
                if k == 0:
                    all_test_labels = labels_test
                else:
                    all_test_labels = np.append(all_test_labels, labels_test, axis = 0)
            
                for i in range(features_test.shape[0]):
                    x_index = int(math.floor(labels_test[i,0] * lateral_partition))
                    if(x_index == lateral_partition): x_index = lateral_partition-1
                    
                    y_index = int(math.floor(labels_test[i,1] * lateral_partition))
                    if(y_index == lateral_partition): y_index = lateral_partition-1
                    
                    true_index = (y_index * lateral_partition) + x_index
                    
                    if true_index == index:
                        correct_class.append(1)
                        correct_distance.append(distance_output[i])
                    else:
                        correct_class.append(0)
                        incorrect_distance.append(distance_output[i])  
            
            correct_percentage = sum(correct_class) / len(correct_class)
            
            if len(correct_distance) > 0:
                avg_corr_distance = sum(correct_distance) / len(correct_distance)
            else:
                avg_corr_distance = 0
            
            if len(incorrect_distance) > 0:
                avg_incorr_distance = sum(incorrect_distance) / len(incorrect_distance)
            else:
                avg_incorr_distance = 0    
            
            print("Test set statistics: {0:.3f}% correct predictions; avg distances - correct = {1:.3f} m, incorrect = {2:.3f} m"
                .format(correct_percentage*100,avg_corr_distance, avg_incorr_distance))
            
            session_name = 'nn_class_' + str(index)
            saver.save(sess, 'class_data/results/' + session_name)   
            
            #checks the results so far
            non_zeros = np.asarray(non_zeros)
            distance_output = np.asarray(distance_output)
            results_so_far(index, distance_output, all_test_labels, non_zeros)  
              
              
        #   Runs the training	  
        ##########################################################
    
    else:
        print("Too little data - attempting to train would break this code :D")
        
        distances = []
        non_zeros = []
        these_labels_x = []
        these_labels_y = []
        
        for k in range(test_sets):
            
            #if there are test samples in there, we assume the system predicts the attributed class center
            labels_test = labels[-(k+1)]
            features_test = features[-(k+1)]
            test_samples = labels_test.shape[0]
            for i in range(test_samples):
                    
                delta_x = (labels_test[i,0] - x_pos) * data_downscale
                delta_y = (labels_test[i,1] - y_pos) * data_downscale
                
                this_distance = math.sqrt(delta_x**2 + delta_y**2)
                
                distances.append(this_distance)
                non_zeros.append(np.sum(features_test[i]))
                these_labels_x.append(labels_test[i,0])
                these_labels_y.append(labels_test[i,1])
        
        if len(distances) > 0:
            distances = np.asarray(distances)  
            labels_test = np.asarray([these_labels_x, these_labels_y])
            non_zeros = np.asarray(non_zeros)
            print("{0} test samples found for this class, with average error of {1}".format(test_samples, np.mean(distances)))
            results_so_far(index, distances, labels_test, non_zeros) 
            
        


def results_so_far(index, new_distances, new_labels, new_non_zeros):
    
    #Double check sizes
    assert new_distances.shape[0] == new_non_zeros.shape[0]
    assert new_distances.shape[0] == new_labels.shape[0]
    
    sim_results_file = 'class_data/results/sim_results'
    
    #Loads the distances and/or updates it
    if os.path.isfile(sim_results_file):
        with open(sim_results_file, 'rb') as f:
                distances, labels, non_zeros = pickle.load(f)
                
        distances = np.append(distances, new_distances)
        labels = np.append(labels, new_labels, axis = 0)
        non_zeros = np.append(non_zeros, new_non_zeros)
    else:
        distances = new_distances
        labels = new_labels
        non_zeros = new_non_zeros
        
    #Stores the new data
    with open(sim_results_file, 'wb') as f:   #<---- TODO: pickle -> dill
        pickle.dump([distances, labels, non_zeros], f)
    
    #Computes and prints up to date metrics
    distances = np.asarray(distances)
    avg_distance = np.mean(distances)
    sorted_distances = np.sort(distances)
    distance_95 = sorted_distances[int(distances.shape[0] * 0.95)]
    
    print("Current results: {0} positions evaluated, avg error (m) = {1:.4f}, 95% percentile (m) = {2:.4f}"
        .format(distances.shape[0], avg_distance, distance_95))
        
        

if __name__ == "__main__":

    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', help="class index to train",
            default=None)
    args = parser.parse_args()
    
    if args.index is None:
        index = 3 # <-- dbg
        print_train = True
        print_periodicity = 1
    else:
        index = int(args.index)
        print_train = False
        print_periodicity = 100
    
    train_a_class(index, print_periodicity, print_train)  
 
    #Prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))