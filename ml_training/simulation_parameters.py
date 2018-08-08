#Simulations parameters file:
#created to ensure consistency throughout ALL other files


##########################################################
#Data parameters   -   From the data, DO NOT CHANGE

data_file = '../data/final_table'
data_downscale = 400    #x and y have a range of 400, scaled down to [0,1]
max_time = 6		    #in micro seconds
sample_freq = 20	    #in MHz
beamformings = 32

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

#Data parameters
##########################################################






##########################################################
#Pre-processing parameters

preprocessed_file = '../data/tf_dataset'

baseline_cut = -100     #in dBm
tx_power = 45           #in dBm
rx_gain = 10            #in dBi

minimum_power = baseline_cut - (tx_power-original_tx_power) - (rx_gain-original_rx_gain)

# stored power if non-zero = (power+power_offset)*power_scale 
#Power offset = 170, scale = 0.01 -> -50dbm in the data becomes 1.2; -150dbm becomes 0.2; 
power_offset = 170
power_scale = 0.01
min_pow_cutoff = ((minimum_power+power_offset) * power_scale)   #minimum_power converted


removed_ts = True
slice_weak_TS_start_remove = 82 #for sample_freq = default = 20MHz
removed_invalid_slots = False
time_slots = max_time * sample_freq
if removed_ts:
    #rescales the TS to remove to the target freq
    slice_weak_TS_start_remove = int((slice_weak_TS_start_remove / 20) * sample_freq)
    time_slots = time_slots - ( (time_slots-slice_weak_TS_start_remove) + 1)   #removed slots [0, slice_weak_TS_start_remove through (max_time * sample_freq)]

#noise STD
test_noise = 6.00    #log-normal distribution = gaussian over dB values
noise_std_converted = (test_noise * power_scale)

#scaler
binary_scaler = True
only_16_bf = False           #<--- change maxpool from [2,1] to [1,1] when this is true, to keep the same resource utilization and thus enable a fair comparison
#todo -> mudar isto devidamente, alterando a variável beamforings
predicted_input_size = time_slots * beamformings

#Misc.
detect_invalid_slots = False
slice_weak_TS = removed_ts
test_spatial_undersampling = 1   # =1 -> 1m between samples, =2 -> 2m, ...
                                 # min = 1 m
#Pre-processing parameters
##########################################################



##########################################################
#CNN - Classification parameters

lateral_partition = 8                       #this will result in 'lateral_partition'^2 classes!!
area_partition = lateral_partition ** 2
n_classes = area_partition                  #this variable has 2 instances, for readability 

dnn_classification_parameters = {   'batch_size': 64,
                                    'epochs': 1000,
                                    'dropout': 0.01,
                                    'learning_rate': 1e-4,
                                    'learning_rate_decay': 0.995,
                                    'fcl_neurons': 256,
                                    'hidden_layers': 12,
                                    'cl_neurons_1': 8,
                                    'cl_filter_1': [3,3],
                                    'cl_maxpool_1': [2,1],  #<--- as currently defined, the "temporal" dimention has 81 slots: 
                                                            #divisible by 1, 3, 9, 27 -> don't try maxpool(2,2), it will crash 
                                    'test_batch_size': 256  }

#CNN - Classification parameters
##########################################################



##########################################################
#CNN - Regression parameters

train_sets = 20
test_sets = 10
n_predictions = train_sets+test_sets   #<-- it's to heavy to generate them at runtime :(

dnn_regression_parameters = {   'batch_size': 64,
                                    'epochs': 1000,
                                    'dropout': 0.01,
                                    'learning_rate': 1e-4,
                                    'learning_rate_decay': 0.995,
                                    'fcl_neurons': 256,
                                    'hidden_layers': 12,
                                    'cl_neurons_1': 8,
                                    'cl_filter_1': [3,3],
                                    'cl_maxpool_1': [2,1],  #<--- as currently defined, the "temporal" dimention has 81 slots: 
                                                            #divisible by 1, 3, 9, 27 -> don't try maxpool(2,2), it will crash 
                                    'test_batch_size': 256  }

#CNN - Regression parameters
##########################################################

