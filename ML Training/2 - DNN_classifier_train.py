######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import time
import tensorflow as tf
import numpy as np
import pickle
import math
from sklearn.utils import shuffle


start = time.time()




#Runs "DNN_definition.py" and keeps its variables           [loads the data and the DNN, which in turn loads the simulation parameters]
exec(open("DNN_definition_class.py").read(), globals())



##########################################################
#   Runs the code

# if n_classes == 1, instead of training the classification, jumps straight into the regression
#                       [because there is only 1 class, with 100% accuracy]
if n_classes == 1:
    print("Training for 1 class - doing regression instead of classification")
else:
    print("Training for {0} classes!".format(n_classes))


#Starts the Session
saver = tf.train.Saver()
with tf.Session() as sess:
  
    if n_classes > 1:
        best_acc = 0.0
    else:
        best_acc = 9999999.0
  
    #initializes the variables
    print("Initializing the variables and starting the training...")
    sess.run(tf.global_variables_initializer())
  
  
    #creates the features
    features_train, labels_train = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler, only_16_bf, undersampling = test_spatial_undersampling)
    if n_classes > 1:
        labels_train = position_to_class(labels_train, lateral_partition)
    train_set_size = features_train.shape[0]
    num_batches = int(math.ceil(train_set_size / batch_size))
    features_train, labels_train = shuffle(features_train, labels_train)
    assert features_train.shape[0] == labels_train.shape[0]
  
    #runs the training phase
    current_batch = 0
    epochs_completed = 0
    epochs_not_improving = 0
    while((epochs_completed < epochs) and (epochs_not_improving < 50)):
    
        
        start_batch = current_batch*batch_size
        if ((current_batch+1)*batch_size) > train_set_size:
            batch = [features_train[start_batch:], 
                    labels_train[start_batch:]]
        else:
            end_batch = ((current_batch+1)*batch_size)
            batch = [features_train[start_batch:end_batch], 
                    labels_train[start_batch:end_batch]]
        
        #every 128 iterations, prints a life signal
        if current_batch % 128 == 0:
          print('.', end = '', flush = True)
        
          
        #trains the network (with dropout)
        if n_classes > 1:
            train_step_c.run(feed_dict={input: batch[0], position: batch[1], keep_prob: 1.0-dropout, learning_rate_var: learning_rate})
        else:
            train_step_l.run(feed_dict={input: batch[0], location: batch[1], keep_prob: 1.0-dropout, learning_rate_var: learning_rate})
        
        
        #If the epoch is finished
        current_batch += 1
        if(current_batch >= num_batches):
        
            #Sets a new train set if noisy data is wanted:
            if noise_std_converted != 0.0:
                features_train, labels_train = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler, only_16_bf, undersampling = test_spatial_undersampling)
                if n_classes > 1:
                    labels_train = position_to_class(labels_train, lateral_partition)
                train_set_size = features_train.shape[0]
                num_batches = int(math.ceil(train_set_size / batch_size))
                features_train, labels_train = shuffle(features_train, labels_train)
                assert features_train.shape[0] == labels_train.shape[0]
            
            
            current_batch = 0
            epochs_completed += 1
            
            
            if n_classes > 1:
                #evaluates the NN on the test set [since the CNN takes a lot of resources, a loop must be used!]
                accuracy_output = []
                end_test = 0
                j = 0
                while(end_test < features_test.shape[0]):
                    start_test = j*test_batch_size
                    end_test = (j+1) *test_batch_size
                    j = j+1
                    
                    if(end_test > features_test.shape[0]):
                        end_test = features_test.shape[0]
                    
                    accuracy_output.append(accuracy.eval(feed_dict={
                        input: features_test[start_test:end_test], position: labels_test[start_test:end_test], keep_prob: 1.0}))
                
                #flattens that list
                accuracy_output = np.mean(accuracy_output)
                print('Finished Epoch {0},   Accuracy= {1},    next LR = {2}'
                    .format(epochs_completed, accuracy_output, learning_rate))
                    
                if accuracy_output > best_acc:
                    epochs_not_improving = 0
                    best_acc = accuracy_output
                else:
                    epochs_not_improving += 1
                    
                    
                
            else:
                #evaluates the NN on the test set [since the CNN takes a lot of resources, a loop mas be used!]
                distance_output = []
                end_test = 0
                j = 0
                while(end_test < features_test.shape[0]):
                    start_test = j*test_batch_size
                    end_test = (j+1) *test_batch_size
                    j = j+1
                    
                    if(end_test > features_test.shape[0]):
                        end_test = features_test.shape[0]
                    
                    distance_output.append(distance.eval(feed_dict={input: features_test[start_test:end_test], 
                        location: labels_test[start_test:end_test], keep_prob: 1.0}))
                
                #flattens that list
                distance_output = [item for sublist in distance_output for item in sublist]
                avg_distance = np.mean(distance_output)
                sorted_distance = np.sort(distance_output)
                distance_95 = sorted_distance[int(features_test.shape[0] * 0.95)]
                
                print('Finished Epoch {0},   Test distance (m)= {1:.4f},   95% percentile (m) = {2:.4f},   next LR = {3:.4e}'
                                .format(epochs_completed, avg_distance, distance_95, learning_rate))
                                
                if avg_distance < best_acc:
                    epochs_not_improving = 0
                    best_acc = avg_distance
                else:
                    epochs_not_improving += 1
            
            
            
            #Adapt the learning rate
            learning_rate = learning_rate * learning_rate_decay
    
    session_name = 'nn_result_' + scaler_name + '_%.2f' % test_noise
    saver.save(sess, 'results/' + session_name)   
	  
	  
	  
#   Runs the code	  
##########################################################


      
#Prints the execution time
end = time.time()
exec_time = (end-start)
print("Execution time = {0:.4}s".format(exec_time))