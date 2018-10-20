'''
DNN_definition_class.py

-> Defines the DNN that will be used in the classification part
   [If the number of classes is set as 1, does the regression instead]
'''

#Runs "load_data.py" and keeps its variables
exec(open("load_data.py").read(), globals())

#Sets the target GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(target_gpu)
print("[Using GPU #{0}]".format(target_gpu))

#Loads hyperparameters from the simulation parameters
dictionary = dnn_classification_parameters
batch_size = dictionary['batch_size']
epochs = dictionary['epochs']
dropout = dictionary['dropout']
learning_rate = dictionary['learning_rate']
learning_rate_decay = dictionary['learning_rate_decay']
fcl_neurons = dictionary['fcl_neurons']
hidden_layers = dictionary['hidden_layers']

cl_neurons_1 = dictionary['cl_neurons_1']
cl_filter_1 = dictionary['cl_filter_1']
cl_maxpool_1 = dictionary['cl_maxpool_1']

test_batch_size = dictionary['test_batch_size']


##########################################################
#   NN Auxiliary functions

def get_fans(shape):
    '''Xavier initialization auxiliary function'''
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def weight_variable(shape, std = None):
    '''Function to initialize the weights as random variables'''
    #if the std is not explicitly added, uses Xavier init
    if std is None:
        fan_in, fan_out = get_fans(shape)
        #TODO: try removing fan_out [https://arxiv.org/pdf/1502.01852.pdf] <----
        std = np.sqrt(2. / (fan_in + fan_out))
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)


def bias_variable(shape, val = 0.01):
    '''Function to initialize the bias. For ReLUs, it MUST be > 0.0'''
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '''2D convolution layer wrapper'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, x, y):
    '''Maxpool layer wrapper'''
    return tf.nn.max_pool(input, ksize=[1, x, y, 1],
                        strides=[1, x, y, 1], padding='SAME')


def batch_norm(x, phase):
    '''Batch normalization wrapper'''
    return tf.layers.batch_normalization(x,center = True, scale = True,
        training = phase)

#   NN Auxiliary functions
##########################################################


 ##########################################################
#   NN Architecture

#Control variables
learning_rate_var = tf.placeholder(tf.float32, shape=[])        # The current learning rate
phase = tf.placeholder(tf.bool)                                 # Bool that tells the system whether it is training (1) or testing (0)
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')      # (1 - Dropout) probability


#Defines the "inputable" data (data + results)
input_size = predicted_input_size
this_bf = 32
if only_16_bf:
    input_size = input_size/2
    this_bf = 16
input = tf.placeholder(tf.float32, shape=[None, input_size], name = 'input')
position = tf.placeholder(tf.int64, shape=[None], name = 'position')   #<--- TODO: rename as "class"
location = tf.placeholder(tf.float32, shape=[None, 2], name = 'location')

#Reshapes "input" so as to be the input of the conv layer
#Reshape format explained: -1 = # of entries per batch; BFxTS image; 1 feature (power)
input_2d = tf.reshape(input, [-1, this_bf, time_slots, 1])


#CONVOLUTION LAYER 1
#weight shape: filter_dim_1, filter_dim_2, 1 feature (power), cl_neurons outputs
W_conv1 = weight_variable([cl_filter_1[0], cl_filter_1[1], 1, cl_neurons_1])
b_conv1 = bias_variable([cl_neurons_1])
h_conv1 = tf.nn.relu(conv2d(input_2d, W_conv1) + b_conv1)
#batch norm
# b_norm_conv1 = batch_norm(h_conv1,phase)
#pooling
# h_pool1 = max_pool(b_norm_conv1, cl_maxpool_1[0], cl_maxpool_1[1]) #<-- w batch norm
h_pool1 = max_pool(h_conv1, cl_maxpool_1[0], cl_maxpool_1[1]) #<-- w/o batch norm
# 2d -> flat
h_pool1_elements = int((cl_neurons_1 * this_bf * time_slots) / (cl_maxpool_1[0] * cl_maxpool_1[1]))
h_pool1_flat = tf.reshape(h_pool1, [-1, h_pool1_elements])


#FULLY CONNECTED LAYER 1  <---- FCL w/o batch norm - I got worse results with batch norm on
W_fc = [weight_variable([h_pool1_elements, fcl_neurons])]
b_fc = [bias_variable([fcl_neurons])]
h_fc = [tf.nn.relu(tf.matmul(h_pool1_flat, W_fc[0]) + b_fc[0])]
h_fc_drop = [tf.nn.dropout(h_fc[0], keep_prob)]


#FULLY CONNECTED LAYER 2 through "hidden_layers"
for i in range(1, hidden_layers):
    W_fc.append( weight_variable([fcl_neurons, fcl_neurons]) )
    b_fc.append( bias_variable([fcl_neurons]) )
    h_fc.append( tf.nn.relu(tf.matmul(h_fc_drop[i-1], W_fc[i]) + b_fc[i]) )
    # bn_fc.append( batch_norm(h_fc[i], phase))
    h_fc_drop.append( tf.nn.dropout(h_fc[i], keep_prob))


#RESULTS LAYER - CLASSIFICATION
if n_classes > 1:
    W_fc_c = weight_variable([fcl_neurons, n_classes])
    b_fc_c = bias_variable([n_classes])
    prediction_c = tf.matmul(h_fc_drop[-1], W_fc_c) + b_fc_c


#RESULTS LAYER - REGRESSION [will be used when n_classes == 1]
else:
    W_fc_l = weight_variable([fcl_neurons, 2], std = 0.01)
    bias_x = bias_variable([1], val = 0.5)
    bias_y = bias_variable([1], val = 0.5)
    b_fc_l = tf.concat([bias_x, bias_y],0)
    prediction_l = tf.matmul(h_fc_drop[-1], W_fc_l) + b_fc_l          # learns with the original value
    prediction_clip = tf.clip_by_value(prediction_l, 0.0, 1.0)        # gets the final results with the clipped version

#   NN Architecture
##########################################################


##########################################################
#   TensorFlow functions

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Add for batch Normalization
with tf.control_dependencies(update_ops):                # Add for batch Normalization

    #Classification
    if n_classes > 1:
        softmax_pred = tf.nn.softmax(prediction_c, name = 'softmax')
        correct_prediction = tf.equal(tf.argmax(prediction_c, 1), position)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

        #defines the loss function (= avg( cross_entropy( delta (target_value - obtained_softmax) ) )
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=position, logits=prediction_c))

        #defines the optimizer (ADAM)
        train_step_c = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(cross_entropy)

    #Regression
    else:
        delta_positions = location - prediction_clip 		            # contains delta_x AND delta_y
        delta_positions_squared = tf.square(delta_positions * data_downscale)
                                                                        # contains delta_x^2 AND delta_y^2, SCALING UP BACK TO THE ORIGINAL RANGE [0;data_downscale]
        distance_square = tf.reduce_sum(delta_positions_squared,1)      # reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
        distance = tf.sqrt(distance_square, name='distance')	        # and then computes its square root

        # loss as MMSE
        loss_function = tf.reduce_mean(tf.square(location - prediction_l))

        # defines the optimizer (ADAM)
        train_step_l = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss_function)

#   TensorFlow functions
##########################################################
