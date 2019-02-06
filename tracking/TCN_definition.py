'''
TCN_definition.py

-> Defines the complete TCN Architecture
'''
import TCN_classes

# Loads the tracking simulation parameters
exec(open("simulation_parameters.py").read(), globals())

# Sets the target GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(target_gpu)

# Loads hyperparameters from the tracking parameters
dictionary = tcn_parameters
batch_size = dictionary['batch_size']
epochs = dictionary['epochs'] * train_split
learning_rate = dictionary['learning_rate']
learning_rate_decay = dictionary['learning_rate_decay']
tcn_layers = dictionary['tcn_layers']
tcn_filter_size = dictionary['tcn_filter_size']
tcn_features = dictionary['tcn_features']
dropout = dictionary['dropout']

# Defines auxiliary variables
print("(TCN memory = {} steps)".format(time_steps))
if isinstance(tcn_features, list):
    features_list = tcn_features
else:
    features_list = [tcn_features] * tcn_layers

input_length = predicted_input_size
output_length = 2


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
#   NN Auxiliary functions
##########################################################


##########################################################
#   NN Architecture

#Control variables
learning_rate_var = tf.placeholder(tf.float32, shape=[])    # The current learning rate
is_training = tf.placeholder("bool", name='is_training')    # Defines whether it is training

# TF Graph input
X = tf.placeholder(tf.bool, [None, time_steps, input_length], name='X')
Y = tf.placeholder(tf.float32, [None, output_length], name='Y')

# Defines the TCN core (check TCN_classes.py)
TCN = TCN_classes.TemporalConvNet(features_list,
                                  kernel_size=tcn_filter_size,
                                  dropout=dropout)

# Prepares the output layer (regression)
W_output = weight_variable([features_list[-1], 2])
bias_x = bias_variable([1], val = 0.5)
bias_y = bias_variable([1], val = 0.5)
b_output = tf.concat([bias_x, bias_y], 0)

prediction = tf.matmul(TCN(tf.cast(X, tf.float32), training=is_training)[:, -1, :],
    W_output) + b_output
prediction_clip = tf.clip_by_value(prediction, 0.0, 1.0)

# prediction = tf.layers.dense(TCN(X, training=is_training)[:, -1, :],
    # output_length, activation=None,
    # kernel_initializer=tf.orthogonal_initializer(),
    # bias_initializer=tf.Variable(tf.constant(0.5, shape=[output_length]))    )
# prediction_clip = tf.clip_by_value(prediction, 0.0, 1.0)

#   NN Architecture
##########################################################


##########################################################
#   TensorFlow functions

#DISTANCE -------------------------------------------------------
# contains series with delta_x AND delta_y (delta being the difference)
delta_position = Y - prediction_clip

# contains delta_x^2 AND delta_y^2, SCALING UP BACK TO THE ORIGINAL RANGE [0;data_downscale]
delta_squared = tf.square(delta_position * data_downscale)

# reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
distance_squared = tf.reduce_sum(delta_squared,1)

# and then computes its square root
distance = tf.sqrt(distance_squared, name='distance')


#LOSS AND TRAIN -------------------------------------------------------
# loss as MMSE (regression)
loss = tf.reduce_mean(tf.square(Y - prediction))

# defines the optimizer (ADAM)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss)

#   TensorFlow functions
##########################################################




##########################################################
##########################################################
##########################################################
#   OLD (some code regarding LSTMs)
"""
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
#   NN Auxiliary functions
##########################################################


##########################################################
#   NN Architecture

#Control variables
learning_rate_var = tf.placeholder(tf.float32, shape=[])  # The current learning rate

#Input/Output data placeholders
#   (shape=[batch_size/None, time_steps, single_element_shape])
# input_size = predicted_input_size
input_size = 2
input_sequence = tf.placeholder(tf.float32, [None, time_steps, input_size])
real_location = tf.placeholder(tf.float32, [None, 2])

#TCN layer definition
cell = tf.nn.tcn_cell.LSTMCell(tcn_neurons)
initial_state = cell.zero_state(batch_size, dtype=tf.float32)
tcn_output, _ = tf.nn.dynamic_tcn(cell, input_sequence,
    initial_state = initial_state)
#tcn_output shape: [batch_size, time_steps, tcn_neurons]
#   the "_", usually called state, is the same as tcn_output[-1] for a single
#   TCN layer.

#MLP Layer 1
assert mlp_layers >= 1
W_mlp = [weight_variable([tcn_neurons, mlp_neurons])]
b_mlp = [bias_variable([mlp_neurons])]
h_mlp = [tf.nn.relu(tf.matmul(tcn_output[:,-1,:], W_mlp[0]) + b_mlp[0])]
#tcn_output[:, -1, :] fetches the final state of the TCN, i.e. its output AFTER
#   going through the sequence

#Other MLP Layers
for i in range(1, mlp_layers):
    W_mlp.append( weight_variable([mlp_neurons, mlp_neurons]) )
    b_mlp.append( bias_variable([mlp_neurons]) )
    h_mlp.append( tf.nn.relu(tf.matmul(h_mlp[i-1], W_mlp[i]) + b_mlp[i]) )

#Regression (output layer)
W_output = weight_variable([mlp_neurons, 2])
bias_x = bias_variable([1], val = 0.5)
bias_y = bias_variable([1], val = 0.5)
b_output = tf.concat([bias_x, bias_y], 0)

prediction = tf.matmul(h_mlp[-1], W_output) + b_output
prediction_clip = tf.clip_by_value(prediction, 0.0, 1.0)

#   NN Architecture
##########################################################


##########################################################
#   TensorFlow functions

#DISTANCE -------------------------------------------------------
# contains series with delta_x AND delta_y
delta_position = real_location - prediction_clip

# contains delta_x^2 AND delta_y^2, SCALING UP BACK TO THE ORIGINAL RANGE [0;data_downscale]
delta_squared = tf.square(delta_position * data_downscale)

# reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
distance_squared = tf.reduce_sum(delta_squared,1)

# and then computes its square root
distance = tf.sqrt(distance_squared, name='distance')


#LOSS AND TRAIN -------------------------------------------------------
# loss as MMSE
loss = tf.reduce_mean(tf.square(real_location - prediction))

# defines the optimizer (ADAM)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss)

#   TensorFlow functions
##########################################################
"""