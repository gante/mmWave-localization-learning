'''
model_definition.py

-> Defines the complete TCN or LSTM Architecture, according to simulation_parameters
'''
import TCN_classes

# Loads the tracking simulation parameters
exec(open("simulation_parameters.py").read(), globals())

# Sets the target GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)

# Defines auxiliary variables
print("(Input length = {} steps)".format(time_steps))
input_length = predicted_input_size
output_length = 2

# Defines Input and Output layers, as well as the TF control variables
#   (which are the same, regardless of the model)
# [Control variables]
learning_rate_var = tf.placeholder(tf.float32, shape=[])    # The current learning rate
is_training = tf.placeholder("bool", name='is_training')    # Defines whether it is training
# [TF Graph I/O]
X = tf.placeholder(tf.bool, [None, time_steps, input_length], name='X')
X_casted = tf.cast(X, tf.float32) #[0, 1] (better results than [-1, 1])
Y = tf.placeholder(tf.float32, [None, output_length], name='Y')


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
#   TCN Architecture

if use_tcn:
    dictionary = tcn_parameters
    tcn_layers = dictionary['tcn_layers']
    tcn_filter_size = dictionary['tcn_filter_size']
    tcn_features = dictionary['tcn_features']
    batch_size = dictionary['batch_size']
    learning_rate = dictionary['learning_rate']
    learning_rate_decay = dictionary['learning_rate_decay']
    dropout = dictionary['dropout']

    if isinstance(tcn_features, list):
        features_list = tcn_features
    else:
        features_list = [tcn_features] * tcn_layers

    # Defines the TCN core (check TCN_classes.py)
    TCN = TCN_classes.TemporalConvNet(features_list,
                                      kernel_size=tcn_filter_size,
                                      dropout=dropout)

    # Prepares the output layer (regression)
    W_output = weight_variable([features_list[-1], 2])
    bias_x = bias_variable([1], val = 0.5)
    bias_y = bias_variable([1], val = 0.5)
    b_output = tf.concat([bias_x, bias_y], 0)

    prediction = tf.matmul(TCN(X_casted, training=is_training)[:, -1, :],
        W_output) + b_output
    prediction_clip = tf.clip_by_value(prediction, 0.0, 1.0)

#   TCN Architecture
##########################################################


##########################################################
#   LSTM Architecture

else:
    dictionary = lstm_parameters
    mlp_layers = dictionary['mlp_layers']
    mlp_neurons = dictionary['mlp_neurons']
    lstm_neurons = dictionary['lstm_neurons']
    batch_size = dictionary['batch_size']
    learning_rate = dictionary['learning_rate']
    learning_rate_decay = dictionary['learning_rate_decay']
    dropout = dictionary['dropout']

    #LSTM layer definition
    cell = tf.nn.rnn_cell.LSTMCell(lstm_neurons)
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    lstm_output, _ = tf.nn.dynamic_rnn(cell, X_casted,
        initial_state = initial_state)
    #lstm_output shape: [batch_size, time_steps, lstm_neurons]

    #MLP Layer 1
    assert mlp_layers >= 1
    W_mlp = [weight_variable([lstm_neurons, mlp_neurons])]
    b_mlp = [bias_variable([mlp_neurons])]
    h_mlp = [tf.nn.relu(tf.matmul(lstm_output[:,-1,:], W_mlp[0]) + b_mlp[0])]
    #tlstm_output[:, -1, :] fetches the final state of the LSTM, i.e. its output AFTER
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

#   LSTM Architecture
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
