#Simulations parameters file:
#created to ensure consistency throughout ALL other files



##########################################################
#Configurable parameters   -   From the data!!!

data_file = 'tf_dataset'
data_downscale = 400    #x and y have a range of 400, scaled down to [0,1]
# data_downscale = 100    #QUICK-TEST MODE: x and y have a range of 100, scaled down to [0,1]
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


#Configurable parameters
##########################################################






##########################################################
#Pre-processing parameters

baseline_cut = -100     #in dBm         -> THIS VALUE SHOULDNT BE CHANGED; assumes 0dBi Rx gain and 30dBm Tx power
tx_power = 45           #in dBm
rx_gain = 10            #in dBi

minimum_power = baseline_cut - (tx_power-30) - (rx_gain)

# stored power if non-zero = (power+power_offset)*power_scale 
#Power offset = 170, scale = 0.01 -> -50dbm becomes 1.2; -150dbm becomes 0.2; 
power_offset = 170
power_scale = 0.01
min_pow_cutoff = ((minimum_power+power_offset) * power_scale)   #minimum_power converted


removed_ts = True
slice_weak_TS_start_remove = 82
removed_invalid_slots = False
if removed_ts:
    time_slots = 120 - ( (120-slice_weak_TS_start_remove) + 1)   #removed slots [0, slice_weak_TS_start_remove-119]
else:
    time_slots = max_time * sample_freq

predicted_input_size = time_slots * beamformings

#noise STD
test_noise = 6.00    #log-normal distribution = gaussian over dB values
noise_std_converted = (test_noise * power_scale)

#scaler
binary_scaler = True

#Pre-processing parameters
##########################################################






##########################################################
#CNN
batch_size = 32
epochs = 1000
dropout = 0.5
learning_rate = 1e-5
learning_rate_decay = 0.99
fcl_neurons = 1024
hidden_layers = 7

cl_neurons_1 = 8
cl_filter_1 = [1,3]
cl_maxpool_1 = [2,1]

test_batch_size = 256

#CNN
##########################################################