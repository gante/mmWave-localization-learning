
import pickle
import numpy as np


#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("simulation_parameters.py").read(), globals())

data_file =  '../' + data_file





#Loads the dataset
print("Loading the dataset...")
with open(data_file, 'rb') as f:
  features, labels, invalid_slots = pickle.load(f)

if(len(invalid_slots) > 0):
    print("WARNING - this dataset has some features removed, for full data preprocess again!")
    
    
    

    
print("Printing the non-zero distribution over the time slots...")
print("(minimum modified power = {0})".format(minimum_power))

total_non_zeros = 0
non_zeros = [0] * time_slots

#for each sample
for i in range(features.shape[0]):
    
    #for each element in the sample
    for j in range(features.shape[1]):
    
        #if it is non-zero, adds to the correct bin
        #DIM 1 = BF, DIM 2 = TS -> lê cada BF seguido
        if(features[i,j] > min_pow_cutoff):
            slot = j % time_slots
            total_non_zeros = total_non_zeros + 1
            non_zeros[slot] = non_zeros[slot] + 1
            
#after all samples have been read, prints the results
cumulative_percentage = 0
for i in range(time_slots):
    percentage = (non_zeros[i] / total_non_zeros)*100
    cumulative_percentage = cumulative_percentage + percentage
    print("SLOT {0}: non-zeros = {1}, percentage of data = {2}, cumulative so far = {3}".format(i, non_zeros[i], percentage, cumulative_percentage))
    

