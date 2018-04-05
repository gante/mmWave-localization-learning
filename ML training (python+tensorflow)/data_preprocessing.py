############################################
#Converts the binary file into a TensorFlow Dataset
############################################

import numpy as np
import pickle
import os
import struct


##########################################################
#   Tunable Things
#From the data:
max_time = 6		#in micro seconds
sample_freq = 20	#in MHz
beamformings = 32
minimum_power = -130    #in dBm

#Tunable parameters
# stored power if non-zero = (power+power_offset)*power_scale 
#Power offset = 170, scale = 0.01 -> -50dbm becomes 1.2; -150dbm becomes 0.2; 
power_offset = 170
power_scale = 0.01

#ALSO SCALES THE POSITIONS INTO [0,1]
starting_x = -183.0
starting_y = -176.0
grid_x = 400.0
grid_y = 400.0
x_shift = 0.0 - starting_x
y_shift = 0.0 - starting_y


#For quick testing (reduces the area, based on [0,1])
quick_test = True
quick_x_min = 0.75
quick_x_max = 1
quick_y_min = 0.5
quick_y_max = 0.75


#Misc.
to_polar = False
detect_invalid_slots = False
slice_weak_TS = True

    
#   Tunable Things    
##########################################################




time_slots = max_time * sample_freq
input_size = time_slots * beamformings
input_w_labels = int(input_size + 2)

data_file = "final_table"




#Loads the binary dataset
print("Loading the binary dataset...")
with open(data_file, mode='rb') as file: # b is important -> binary
    data_binary = file.read()
    
    
    
    
#Converts the binary dataset into float 32
print("Converting the binary to float_32...")
binary_size = os.path.getsize(data_file)
num_elements = int(binary_size/4)
data = struct.unpack('f'*num_elements, data_binary)

del data_binary
del file
  
  

#Converts the dataset into features/labels
print("Converting the dataset into features/labels ...")
num_positions = num_elements / (input_w_labels)
features = []
labels = []

for i in range(int(num_positions)):

    tmp_features = []
    tmp_labels = []
    
    for j in range(input_w_labels):
        item = data[(i * input_w_labels) + j]
        if j == 0:
            item += x_shift
            item /= (grid_x)
            tmp_labels.append(item)
        elif j == 1:
            item += y_shift
            item /= (grid_y)
            tmp_labels.append(item)
        else:
            if(item < 0 and item > minimum_power):
                tmp_features.append((item+power_offset) * power_scale)
            else:
                tmp_features.append(0.0)
            
    features.append(tmp_features)
    labels.append(tmp_labels)
    
    
del data
    
#Converting the features/labels into numpy arrays
print("Converting the features/labels into numpy arrays ...")
features = np.array(features)
labels = np.array(labels)
    
    

#If to polar: (X,Y) -> (R,teta)
if to_polar:
    print("Converting labels into polar coordinates ...")
    
    #x,y = [0,1] -> (rescale) -> x,y = [-0.5, 0.5] -> max r = sqrt(0.25+0.25) -> r = [0,1]
    x = labels[:,0] - 0.5
    y = labels[:,1] - 0.5

    rmax = np.sqrt(0.5)
    r = np.sqrt(np.square(x[:]) + np.square(y[:]))
    r /= rmax
    
    #atan2 = [-pi,pi] -> [0,1]
    tetamax = np.pi
    teta = np.arctan2(y[:],x[:])
    teta = (teta + tetamax) / (2*tetamax)
    
    labels[:,0] = r[:]
    labels[:,1] = teta[:]
    
    print("r range:", labels[:,0].min(), labels[:,0].max())
    print("teta range:", labels[:,1].min(), labels[:,1].max())
    print("power range:", features[:].min(), features[:].max())
else:    
    print("x range:", labels[:,0].min(), labels[:,0].max())
    print("y range:", labels[:,1].min(), labels[:,1].max())
    print("power range:", features[:].min(), features[:].max())

 
#Removing "weak" time_slots [always deletes slots 0 + 82-119]
if slice_weak_TS:
    print("[REMOVING WEAK TS ON:] Removing the specific time slots... ")
    
    mask = np.ones(time_slots*beamformings, dtype=bool)
    
    ts_to_delete = [0]
    
    for i in range((119-83) +1):
        ts_to_delete.append(83 + i)
    
    print("Slots to remove:", ts_to_delete)
    
    for i in range(time_slots*beamformings):
        #DIM 1 = BF, DIM 2 = TS -> lê cada BF seguido
        
        if (i % time_slots) in ts_to_delete:
            mask[i] = False

    #removes those slots from the data
    print("Before TS reduction: ", features.shape)
    features = features[:,mask]
    print("After TS reduction: ", features.shape)
          
    

 
#Detecting invalid slots [i.e. deletes all for this dataset]
invalid_slots = []
if detect_invalid_slots:
    print("[REMOVE ALL USELESS COLUMNS ON:] Detecting the invalid slots",end='',flush=True)
    non_zeros = [0]*time_slots*beamformings
    
    #counts the non-zeroes
    for i in range(features.shape[0]):
        
        if(i % 10000 == 0) and (features.shape[0] > 10000):
            print(".",end='',flush=True)
            
        for j in range(time_slots*beamformings):
            if(features[i,j] > 0):
                non_zeros[j] += 1
    
    #checks which slots have no data
    
    mask = np.ones(time_slots*beamformings, dtype=bool)
    for j in range(time_slots*beamformings):
        if(non_zeros[j] == 0):
            invalid_slots.append(j)
            mask[j] = False
    
    print(" {0} invalid slots out of {1}".format(len(invalid_slots), time_slots*beamformings))
    
    #removes those slots from the data
    features = features[:,mask]
    
    
    
#Quick Tests - If a data reduction is wanted (for faster prototyping of the DNNs), reduces it here
if quick_test:
    print("[AREA REDUCTION ON:] Reducing the area... ")
    
    mask = np.ones(features.shape[0], dtype=bool)
    removed_pos = 0
    
    #Flags the unwanted positions
    for i in range(features.shape[0]):
        
        x = labels[i,0]
        y = labels[i,1]
        
        if((x < quick_x_min) or (x > quick_x_max)):
            mask[i] = False
        else:
            if((y < quick_y_min) or (y > quick_y_max)):
                mask[i] = False
        
    #Removes the unwanted positions
    features = features[mask,:]
    labels = labels[mask,:]
    
    #Reprints the new range
    print("[print features pre-rescaling]")
    print("x range:", labels[:,0].min(), labels[:,0].max())
    print("y range:", labels[:,1].min(), labels[:,1].max())
    print("power range:", features[:].min(), features[:].max())
    
    #Rescales features
    x_range = quick_x_max - quick_x_min
    y_range = quick_y_max - quick_y_min
    for i in range(features.shape[0]):
        labels[i,0] = (labels[i,0] - quick_x_min) / x_range
        labels[i,1] = (labels[i,1] - quick_y_min) / y_range
        
    print("[print features post-rescaling]")
    print("x range:", labels[:,0].min(), labels[:,0].max())
    print("y range:", labels[:,1].min(), labels[:,1].max())
    print("power range:", features[:].min(), features[:].max())
    
    
    
    
#Delecting invalid users
print("Detecting the invalid positions... ",end='',flush=True)
mask = np.ones(features.shape[0], dtype=bool)
removed_pos = 0

for i in range(features.shape[0]):
    if sum(features[i,:]) == 0:
        mask[i] = False
        removed_pos += 1
        
features = features[mask,:]
labels = labels[mask,:]
print("{0} positions removed.".format(removed_pos))




#Final data reports
print("Usable positions = {0}".format(features.shape[0]))

if detect_invalid_slots:
    print("AVG Sparsity = {0}".format(sum(non_zeros) / (features.shape[0]*time_slots*beamformings)))


#Storing the result
print("Storing the result ...")
with open('tf_dataset', 'wb') as f:
    pickle.dump([features,labels, invalid_slots], f)