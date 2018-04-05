



#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("simulation_parameters.py").read(), globals())


#Runs "load_data.py" and keeps its variables                [data loading]
exec(open("load_data.py").read(), globals())



##########################################################
#   NN Architecture

learning_rate_var = tf.placeholder(tf.float32, shape=[])


#defines function to initialize the weights as random variables
def weight_variable(shape, std = 0.1):
  initial = tf.truncated_normal(shape, stddev=std)
  return tf.Variable(initial)

def bias_variable(shape, val = 0.0):
  initial = tf.constant(val, shape=shape)
  return tf.Variable(initial)
  
  
#defines Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def max_pool(input, x, y):
    return tf.nn.max_pool(input, ksize=[1, x, y, 1],
                        strides=[1, x, y, 1], padding='SAME')
                        
                        
                        
                        


#defines the "inputable" data (data + results)
input = tf.placeholder(tf.float32, shape=[None, input_size], name = 'input')
position = tf.placeholder(tf.float32, shape=[None, 2], name = 'position')


#reshapes "input" so as to be the input of the conv layer
input_2d = tf.reshape(input, [-1, beamformings, time_slots, 1]) # "-1" = # of entries per batch; BFxTS image; 1 feature (power)



#CONVOLUTION LAYER 1 - processes BFxTS image with #cl_neurons 1x3 filters (whose weights are W and bias is b), once per "position".
#                       Then, passes each result through a ReLU, resulting in #cl_neurons "BFxTS" features per entry.
#                       Then, reduces them into #cl_neurons "BF/8 x TS" features using a 8x1 pooling.

W_conv1 = weight_variable([cl_filter_1[0], cl_filter_1[1], 1, cl_neurons_1]) #5x5 convolution, 1 feature (power), #cl_neurons outputs
b_conv1 = bias_variable([cl_neurons_1])

h_conv1 = tf.nn.relu(conv2d(input_2d, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, cl_maxpool_1[0], cl_maxpool_1[1])

# 2d -> flat
h_pool1_elements = int((cl_neurons_1 * beamformings * time_slots) / (cl_maxpool_1[0] * cl_maxpool_1[1]))
h_pool1_flat = tf.reshape(h_pool1, [-1, h_pool1_elements])



#CONVOLUTION LAYER 2 - similar to above

# W_conv2 = weight_variable([cl_filter_2[0], cl_filter_2[1], cl_neurons_1, cl_neurons_2]) #5x5 convolution, 1 feature (power), #cl_neurons outputs
# b_conv2 = bias_variable([cl_neurons_2])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool(h_conv2, cl_maxpool_2[0], cl_maxpool_2[1])

# 2d -> flat
# h_pool2_elements = int((cl_neurons_2 * h_pool1_elements) / (cl_maxpool_2[0] * cl_maxpool_2[1] * cl_neurons_1))
# h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_elements])


#FULLY CONNECTED LAYER 1

# W_fc1 = weight_variable([input_size, fcl_neurons])
# b_fc1 = bias_variable([fcl_neurons])
# h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + b_fc1)

W_fc = [weight_variable([h_pool1_elements, fcl_neurons])]
b_fc = [bias_variable([fcl_neurons])]
h_fc = [tf.nn.relu(tf.matmul(h_pool1_flat, W_fc[0]) + b_fc[0])]


#FULLY CONNECTED LAYER 2 through "hidden_layers"
for i in range(1, hidden_layers):
    W_fc.append( weight_variable([fcl_neurons, fcl_neurons]) )
    b_fc.append( bias_variable([fcl_neurons]) )
    h_fc.append( tf.nn.relu(tf.matmul(h_fc[i-1], W_fc[i]) + b_fc[i]) )


#DROPOUT - (Into the last layer)
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
h_fc_r_drop = tf.nn.dropout(h_fc[hidden_layers-1], keep_prob)


#RESULTS LAYER - MEAN
W_fc_u = weight_variable([fcl_neurons, 2])
b_fc_u = bias_variable([2], val = 0.5)

position_u = tf.matmul(h_fc_r_drop, W_fc_u) + b_fc_u


#   NN Architecture
##########################################################







##########################################################
#   TensorFlow functions

#Function to find the distance = sqrt( delta_x^2  + delta_y^2 ) = euclidian norm
delta_positions = position - position_u 		                # contains delta_x AND delta_y
delta_positions_squared = tf.square(delta_positions * data_downscale) 	    
                                                                # contains delta_x^2 AND delta_y^2, SCALING BACK TO THE ORIGINAL RANGE [0;data_downscale]
distance_square = tf.reduce_sum(delta_positions_squared,1)      # reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
distance = tf.sqrt(distance_square, name='distance')	        # and then computes its square root



# loss as MMSE
loss_function = tf.reduce_mean(tf.square(position - position_u))
    
#defines the optimizer (ADAM)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss_function)

#   TensorFlow functions
##########################################################


