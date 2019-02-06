'''
DNN_definition_regr.py

-> Defines the DNN that will be used in the hierarchical regression part
-> At this stage, we will have 1 DNN regressor PER CLASS (the number of
    classes was defined at the "DNN_classifier" stage). This means that each
    regressor trains with data that was predicted as being in a specific class,
    which makes the regressor specialized in that class.
-> Furthermore, after the convolution layer, the softmax output of the
    DNN_classifier is also added as input to the following layer. This means
    that the specialized regressor will be able to see the "class probability
    distribution" for each sample, which enables enhanced estimations.
'''

#Runs "simulation_parameters.py" and keeps its variables
exec(open("simulation_parameters.py").read(), globals())

#Sets the target GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(target_gpu)
print("[Using GPU #{0}]".format(target_gpu))

#Loads hyperparameters from the simulation parameters
dictionary = dnn_regression_parameters
batch_size = dictionary['batch_size']
epochs = dictionary['epochs']
dropout = dictionary['dropout']
lr = dictionary['learning_rate']
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
location = tf.placeholder(tf.float32, shape=[None, 2], name = 'location')
softmax_predictions = tf.placeholder(tf.float32, shape=[None, n_classes], name = 'softmax_predictions')

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

#IMPORTANT: the next line concatenates to ouput of the Conv Layer above with
#   the softmax predictions of the classification DNN!
data_in_fcl = tf.concat([h_pool1_flat, softmax_predictions], 1)
elements_in = h_pool1_elements + n_classes

W_fc = [weight_variable([elements_in, fcl_neurons])]
b_fc = [bias_variable([fcl_neurons])]
h_fc = [tf.nn.relu(tf.matmul(data_in_fcl, W_fc[0]) + b_fc[0])]
h_fc_drop = [tf.nn.dropout(h_fc[0], keep_prob)]


#FULLY CONNECTED LAYER 2 through "hidden_layers"
for i in range(1, hidden_layers):
    W_fc.append( weight_variable([fcl_neurons, fcl_neurons]) )
    b_fc.append( bias_variable([fcl_neurons]) )
    h_fc.append( tf.nn.relu(tf.matmul(h_fc_drop[i-1], W_fc[i]) + b_fc[i]) )
    h_fc_drop.append( tf.nn.dropout(h_fc[i], keep_prob))

#The rest of the functions depend on the last layer, which depends on the selected class

#   NN Architecture
##########################################################
