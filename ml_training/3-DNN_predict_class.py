######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import tensorflow as tf
import pickle
import numpy as np
import math


def split_data(predicted_class, softmax_output, features_test, labels_test, compress_features = True):

    existing_classes = np.max(predicted_class) + 1
    
    for i in range(existing_classes):
        #gets the data for this file, before actually writting it
    
        file_name = 'class_data/class_' + str(i)
        
        this_features = []
        this_softmax = []
        this_labels = []
        
        for j in range(predicted_class.shape[0]):
        
            if i == predicted_class[j]:
                #python's append is much faster than numpy's append
                softmax_batch_index = math.floor(j/256)  #<--- todo: test_batch_size = 256
                softmax_batch = softmax_output[softmax_batch_index] 
                this_softmax.append(softmax_batch[j%256])
                
                this_features.append(features_test[j])
                this_labels.append(labels_test[j])
        
        #after all data has been gathered, appends to the file
        this_softmax = np.asarray(this_softmax)
        this_labels = np.asarray(this_labels)
        
        if compress_features:
            this_features = np.asarray(this_features, dtype = np.bool_)
        else:
            this_features = np.asarray(this_features)
        
        with open(file_name, 'ab') as f:   #<---- TODO: pickle -> dill
            pickle.dump([this_features, this_labels, this_softmax], f)
        




##########################################################
#TODO: there are better ways to load functions :v

#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("simulation_parameters.py").read(), globals())

#Runs "load_data.py" and keeps its variables                [data loading]
exec(open("load_data.py").read(), globals())

test_batch_size = dnn_classification_parameters['test_batch_size']

##########################################################
#   Runs the code

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    
    # defines the saver (aka loader)
    session_name = 'nn_result_' + scaler_name + '_%.2f' % test_noise
    saver = tf.train.import_meta_graph('results/' + session_name + '.meta')
    saver.restore(sess, 'results/' + session_name)
    
    # Redefines the needed stuff [i.e. placeholders AND operations, that must be saved by name]
    graph = tf.get_default_graph()
    
    input = graph.get_tensor_by_name("input:0")
    position = graph.get_tensor_by_name("position:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    
    accuracy = graph.get_tensor_by_name("accuracy:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    
    #now that the session is loaded, predicts stuff!
    
    
    #evaluates the NN on the test set [since the CNN takes a lot of resources, a loop must be used!]
    for i in range(n_predictions):
    
        print("Predicting classes for sampled dataset #{0}...".format(i))
    
        accuracy_output = []
        softmax_output = []
        predicted_class = []
    
        #When undersampling:
        # The test set will never be undersampled, as we want to assess the
        # impact of the dataset resolution on the predictive capabilities
        if i < train_sets:
            features_test, labels_test = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler, only_16_bf, undersampling = test_spatial_undersampling) 
        else:
            features_test, labels_test = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler, only_16_bf)
            
        labels_test_class = position_to_class(labels_test, lateral_partition)
        
        end_test = 0
        j = 0
        while(end_test < features_test.shape[0]):
            start_test = j*test_batch_size
            end_test = (j+1) *test_batch_size
            j = j+1
            
            if(end_test > features_test.shape[0]):
                end_test = features_test.shape[0]
            
            accuracy_output.append(accuracy.eval(feed_dict={
                input: features_test[start_test:end_test], position: labels_test_class[start_test:end_test], keep_prob: 1.0}))
             
            this_softmax = softmax.eval(feed_dict={input: features_test[start_test:end_test], keep_prob: 1.0})
            softmax_output.append(this_softmax)
            
            this_argmax = np.argmax(this_softmax, axis = 1)
            predicted_class = np.append(predicted_class, this_argmax)
            
        #Splits the predictions by predicted class, and appends the results to the selected file
        print("Splitting the resulting data...")
        predicted_class = np.asarray(predicted_class, dtype = 'int32')
        assert features_test.shape[0] == predicted_class.shape[0]
        split_data(predicted_class, softmax_output, features_test, labels_test, compress_features = binary_scaler)
        # ^   compresses the features (float32 -> bool) if the binary scaler is used

            
#Prints metrics
avg_accuracy = np.mean(accuracy_output)
print('\n\nTest Accuracy= {0:.4f}'.format(avg_accuracy*100))
   