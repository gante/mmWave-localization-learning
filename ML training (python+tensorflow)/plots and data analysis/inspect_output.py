
import pickle
import numpy as np


#############################################
#Configurable parameters

#From the data:
data_file = 'tf_dataset'
max_time = 6		#in micro seconds
sample_freq = 20	#in MHz
beamformings = 32

removed_ts = True
slice_weak_TS_start_remove = 82
#removed_invalid_slots = False
if removed_ts:
    time_slots = 120 - ( (120-slice_weak_TS_start_remove) + 1)   #removed slots [0, slice_weak_TS_start_remove-119]
else:
    time_slots = max_time * sample_freq

starting_x = -183.0
starting_y = -176.0
grid_x = 400.0
grid_y = 400.0
grid = [grid_x, grid_y]
x_shift = 0.0 - starting_x
y_shift = 0.0 - starting_y
shift = [x_shift, y_shift]

minimum_power = -130    #in dBm
power_offset = 170
power_scale = 0.01


#Options
plot_samples = False
count_nonzero = True
plot_average_value = False
dataset_is_compressed = False

process_all = True
images_to_process = 10

#histogram_to_plot = -2 -> doesn't plot; = -1 -> plot all
histogram_to_plot = -2

#noise STD
test_noise = 0.01    #log-normal distribution = gaussian over dB values
noise_std_converted = (test_noise * power_scale)
min_pow_cutoff = ((minimum_power+power_offset) * power_scale)   #minimum_power converted

#Configurable parameters
#############################################



#############################################
#Functions

def plot_a_sample(index_to_plot, time_slots, beamformings, grid, shift):

    data = features[index_to_plot]

    if dataset_is_compressed:
        new_data = [0] * (time_slots*beamformings)
        used_data = 0
        for a in range(time_slots*beamformings):
            if a not in invalid_slots:
                new_data[a] = data[used_data]
                used_data += 1
                
        data = np.array(new_data)
        

    features_2d = data.reshape(beamformings, time_slots)
            
    fig, ax = plt.subplots()
            
    X = (labels[index_to_plot][0] * grid[0]) - shift[0]
    Y = (labels[index_to_plot][1] * grid[1]) - shift[1]
            
    title = 'X=' + str(X) + ' Y=' + str(Y)
    ax.set_title(title)
    ax.set_xlabel('Beamforming Index')
    ax.set_ylabel('Sample Number')
            
    cax = plt.imshow(np.transpose(features_2d))
    cbar = fig.colorbar(cax, ticks=[0, 0.2, 0.7, 1.2])
    cbar.ax.set_yticklabels(['no_data', '-150dBm', '-100dBm', '-50dBm'])
    plt.show() #PLOTS things
    

def plot_average(data, time_slots, beamformings):

    features_2d = data.reshape(time_slots, beamformings)
            
    fig, ax = plt.subplots()
            
    title = 'Average Received Power for all Locations (in dBm)'
    ax.set_title(title)
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Beamforming Index')
            
    cax = plt.imshow(np.transpose(features_2d))
    cbar = fig.colorbar(cax, ticks=[-200, -150, -100, -50])
    cbar.ax.set_yticklabels(['-200dBm', '-150dBm', '-100dBm', '-50dBm'])
    plt.show() #PLOTS things
    
    
def plot_histogram(data, histogram_to_plot):

    
    n, bins, patches = plt.hist(data, 50, facecolor='g', alpha=0.75)

    plt.xlabel('Feature Value')
    plt.ylabel('Occorences')
    title = 'Histogram of index:' + str(histogram_to_plot)
    plt.title(title)
    plt.axis([0, max(data)+0.01, 0, max(n)])
    plt.grid(True)
    plt.show()   
    
    
    
def create_noisy_features(features, noise_std_converted, min_pow_cutoff, scaler = None):

    #Quick exit when no noise is needed
    if(noise_std_converted == 0.0):
        if scaler is not None:
            noisy_features = features
            noisy_features = scaler.fit_transform(noisy_features)
        return(noisy_features)

        
    noise = np.random.normal(scale = noise_std_converted, size = features.shape)
    noisy_features = features + noise
    noisy_features[noisy_features < min_pow_cutoff] = 0
    
    # test_1 = features[features > 0]
    # test_2 = noisy_features[noisy_features > 0]
    
    # print(test_1)
    # print(test_2)
    
    # print(test_1.shape)
    # print(test_2.shape)
    
    #removes the entries containing only 0
    mask = np.ones(features.shape[0], dtype=bool)
    for i in range(features.shape[0]):
        
        this_samples_sum = np.sum(noisy_features[i,:])
        if this_samples_sum < 0.01:
            mask[i] = False
            
    noisy_features = noisy_features[mask,:]
    
    #doublecheck
    print(noisy_features.shape)
    
    #Applies the preprocessing, if needed
    if scaler is not None:
        noisy_features = scaler.fit_transform(noisy_features)
    
    return(noisy_features)
    
#Functions
#############################################


#Loads the dataset
print("Loading the dataset...")
with open(data_file, 'rb') as f:
  features, labels, invalid_slots = pickle.load(f)
  
features_size = features.shape[1]

if(len(invalid_slots) > 0):
    print("WARNING - this dataset has some features removed, for full data preprocess again!")


#initializes stuff
if test_noise > 0.0:
    features = create_noisy_features(features, noise_std_converted, min_pow_cutoff)
    
if process_all:
    images_to_process = features.shape[0]
    index = range(images_to_process)
else:
    index = np.random.randint(features.shape[0], size=images_to_process)

if plot_average_value:
    average_mw = [0]*time_slots*beamformings

if count_nonzero:
    non_zeros = [0]*time_slots*beamformings
    
if histogram_to_plot > -1:
    histogram_data = []
elif histogram_to_plot == -1:
    histogram_data = [[] for i in range(features_size)]


if plot_samples or plot_average_value or (histogram_to_plot > -2):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg


#Main loop (1 iteration per data point):
print('Processing {0} data points...'.format(images_to_process))

for i in range(images_to_process):

    if(i % 10000 == 0) and images_to_process > 10000:
        print("data points processed:",i)
   
    if plot_samples:
        plot_a_sample(index[i], time_slots, beamformings, grid, shift)
        
    if count_nonzero:  
        for j in range(time_slots*beamformings):
            if(features[index[i],j] > 0):
                non_zeros[j] += 1
                
                
    if plot_average_value:
        for j in range(time_slots*beamformings):
            if(features[index[i],j] > 0):
                #converts dbm to mw and adds to the accumulator
                power_dbm = (features[index[i],j] / power_scale) - power_offset
                power_mw = 10**(power_dbm/10)
                average_mw[j] = average_mw[j] + power_mw
    
    if histogram_to_plot > -1:
        data = features[index[i],histogram_to_plot]
        if(data > 0):
            histogram_data.append(data)
    elif histogram_to_plot == -1:
        for j in range(features_size):
            data = features[index[i],j]
            if(data > 0):
                histogram_data[j].append(data)
        
    
    
    
#Final processing   
if count_nonzero:
    total_non_zeros = sum(non_zeros)
    percent = (total_non_zeros*100)/(time_slots*beamformings*images_to_process)
    avg = (percent/100) * (time_slots*beamformings)
    print("Non-zeroes = {0} out of {1} ({2}%, avg={3})".format(total_non_zeros, (time_slots*beamformings*images_to_process), percent, avg))
    
    
    for j in range(time_slots*beamformings):
        if(non_zeros[j] == 0):
            ts = j % time_slots
            bf = int( (j - ts) / time_slots)
            print("Not a single valid entry at TS: {0} and BF: {1}".format(ts, bf))
            
    
    
if plot_average_value:
    #for each entry, averages the result (divides by total samples) and converts to dbm
    average_mw = np.array(average_mw)
    average_dbm = average_mw/images_to_process
    average_dbm = 10 * np.log10(average_dbm)
    
    plot_average(average_dbm, time_slots, beamformings)

    
if histogram_to_plot > -1:
    plot_histogram(histogram_data, histogram_to_plot)
elif histogram_to_plot == -1:
    for j in range(features_size):
        plot_histogram(histogram_data[j], j)