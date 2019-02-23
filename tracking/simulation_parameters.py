'''
simulation_parameters.py

-> created to ensure consistency throughout ALL other files
'''


##########################################################
# Data parameters

#From the data, do not change unless you played with the preprocessing part
data_file = '../data_preprocessing/data_processed/final_table'
data_downscale = 400    #x and y have a range of 400, scaled down to [0,1]
max_time = 6		    #in micro seconds
sample_freq = 20	    #in MHz
beamformings = 32

#Parameters obtained from the ray-tracing simulator, DO NOT CHANGE
starting_x = -183.0
starting_y = -176.0
grid_x = 400.0
grid_y = 400.0
grid = [grid_x, grid_y]
x_shift = 0.0 - starting_x
y_shift = 0.0 - starting_y
shift = [x_shift, y_shift]
original_tx_power = 30  #in dBm
original_rx_gain = 0    #in dBi

# Data parameters
##########################################################


##########################################################
# Pre-processing parameters

#Logic behind these variables: the simulations were done with the above
#   parameters. If we want to slightly change our target simulation power
#   levels without re-doing the ray-tracing part, we can it adapt here. The
#   variable "minimum_power" represents the detection threshold
#   ("baseline_cut") scaled by the a posteriori changes. In practical terms,
#   any sampled power below "minimum_power" is below the detection threshold,
#   and thus not detected.
baseline_cut = -100     #in dBm <--- change for different sample_freq (must be > thermal noise)
tx_power = 45           #in dBm
rx_gain = 10            #in dBi
minimum_power = baseline_cut - (tx_power-original_tx_power) - (rx_gain-original_rx_gain)

#Stored power feature if non-zero = (simulated_power[dBm]+power_offset)*power_scale
#E.g.: Power offset = 170, scale = 0.01 ->  -50dbm in the data becomes 1.2;
#                                           -150dbm becomes 0.2;
#Because this is done manually, the input feature's values will have a
#   non-neglegible gap between absence of data (represented as 0) and a barely
#   detectable power_sample (which should be bigger than 0.1), which helps
#   the learning mechanism.
power_offset = 170
power_scale = 0.01
min_pow_cutoff = ((minimum_power+power_offset) * power_scale)   #minimum_power converted

#If you have more than 1 GPU, you can change the target here
target_gpu = 0
preprocessed_file = 'processed_data/tf_dataset'

#[Optional] Remove contiguous time slots with little to no data. Requires
#   hand-tunning for frequencies != 20 MHz.
assert sample_freq == 20, "Please edit this and the following code lines if other sample_freq was selected!"
removed_ts = True
slice_weak_TS_start_remove = 82
removed_invalid_slots = False
time_slots = max_time * sample_freq
if removed_ts:
    #rescales the TS to remove to the target freq
    slice_weak_TS_start_remove = int((slice_weak_TS_start_remove / 20) * sample_freq)
    time_slots = time_slots - ( (time_slots-slice_weak_TS_start_remove) + 1)
    #removed slots: [0, slice_weak_TS_start_remove through (max_time * sample_freq)]

#noise STD
test_noise = 8.00    #log-normal distribution = gaussian over dB values
noise_std_converted = (test_noise * power_scale)

#scaler
binary_scaler = True    #<--- will use a Normalizer if false  [warning: Normalizer needs some code refactoring!]
predicted_input_size = time_slots * beamformings

#Misc. [for other tests]
detect_invalid_slots = False
slice_weak_TS = removed_ts
spatial_undersampling = 1       # =1 -> 1m between samples, =2 -> 2m, ...
assert spatial_undersampling >= 1, "spatial_undersampling cannot be smaller " \
    "than 1, due to dataset constraints"
# Pre-processing parameters
##########################################################


##########################################################
# Tracking parameters

# -> path options: 's' for static, 'p' for pedestrian, 'c' for car
tracking_folder = 'tracking_data/'
path_file_train = tracking_folder + 'paths_train'
path_file_valid = tracking_folder + 'paths_valid'
path_file_test = tracking_folder + 'paths_test'
path_options = {'s_paths': True,            # enables static paths

                'p_paths': True,            # enables pedestrian-like paths
                'p_avg_speed': 1.4,         # pedestrian avg speed (m/s)[ https://en.wikipedia.org/wiki/Preferred_walking_speed ]
                'p_max_speed': 2.0,         # pedestrian max speed (m/s)
                'p_speed_adjust': 0.3,      # pedestrian max speed adjust (m/s^2)
                'p_direction_adjust': 10.0, # pedestrian max direction adjust (angle in degrees)
                #pedestrian probability of [no change; full stop; direction adjust; speed adjust] each second
                'p_move_proba': [0.8, 0.1, 0.05, 0.05],

                'c_paths': True,            # enables car-like paths
                'c_avg_speed': 8.3,         # avg speed (m/s): ~30kmh or ~18.6mph
                'c_max_speed': 13.9,        # max speed (m/s): ~50kmh or ~31.1mph
                'c_speed_adjust': 3,        # car max speed adjust (m/s^2)
                'c_direction_adjust': 5.0,  # car max direction adjust (angle in degrees)
                #car probability of [no change; full stop; direction adjust; speed adjust] each second
                'c_move_proba': [0.8, 0.02, 0.05, 0.13],
                }
assert sum(path_options['p_move_proba']) == 1.0
assert sum(path_options['c_move_proba']) == 1.0

moving_paths_multiplier = 4.0  #Samples "N*number_of_static_paths" moving paths

parallel_jobs = 2       #uses multiple threads to obtain sequence samples :D
                        #[best results with 2 jobs in CPUs w/ hyperthreading,
                        #   it's a memory bound task]
train_split = 10        #splits an epoch in N sub-epochs (min = 1),
                        #to alliviate RAM requirements
train_split *= ((2.0 * moving_paths_multiplier) + 1.0) / 3.0
train_split = int(train_split)      #Adjusts the split size acording to the number of sampled moving paths

#test & valid length / RAM usage options
valid_size = 0.1  #Defines the validation set size, relative to the train set size
test_size = 1.0             #Big test size = good generalization assessment
n_tests = 2                 #Avg number of times it evaluates a given test path
test_split = train_split

# The train script evaluates the validation set at every "K*train epoch", where a
#   "train epoch" is "100.0/train_split"% of an actual epoch over the train set.
# While it is somewhat expensive, allows us to have some neat control over
#   the "early stopping". Here we define the early stopping as a function of
#   "train epochs": if the validation accuracy does not improve over N "K*train
#   epochs", the training is complete.
valid_assessment_period = 5
early_stopping = int(50.0 / float(valid_assessment_period))
epochs_hard_cap = train_split * 100  # ~ 8 hours on my system. After epoch 500, the
                                     #   improvement is marginal

# Tracking parameters
##########################################################



use_tcn = True #<----------------------------------------------------------------- TCN/LSTM toggle

##########################################################
# TCN parameters
if use_tcn:
    tcn_parameters = {  'batch_size': 64,
                        'learning_rate': 5e-4,
                        'learning_rate_decay': 0.995,
                        'tcn_layers': 2,
                        'tcn_filter_size': 3,
                        'tcn_features': 512,
                        'dropout': 0.0,
                     }

    time_steps = 7

# TCN parameters
##########################################################


##########################################################
# LSTM parameters
else:
    lstm_parameters = { 'batch_size': 64,
                        'learning_rate': 5e-5,
                        'learning_rate_decay': 0.995,
                        'mlp_layers': 3,
                        'mlp_neurons': 512,
                        'lstm_neurons': 512,
                        'dropout': 0.0,
                      }

    time_steps = 7

# LSTM parameters
##########################################################