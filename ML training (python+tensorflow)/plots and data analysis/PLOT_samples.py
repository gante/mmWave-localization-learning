
import pickle
import numpy as np
import matplotlib.pyplot as plt


#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("../simulation_parameters.py").read(), globals())

data_file =  '../' + data_file



#############################################
#Functions

def plot_a_sample(index_to_plot, time_slots, beamformings):

    data = features[index_to_plot]

    #May be needed when non-CNN are used
    # if dataset_is_compressed:
        # new_data = [0] * (time_slots*beamformings)
        # used_data = 0
        # for a in range(time_slots*beamformings):
            # if a not in invalid_slots:
                # new_data[a] = data[used_data]
                # used_data += 1
                
        # data = np.array(new_data)
        
        
    #Adjusts for increased power and rx gain (when compared to the baseline)
    features_adjust = (baseline_cut - minimum_power) * power_scale
    print("Adjusting {0} dB...".format(features_adjust/power_scale), end = '')
    
    data[data > 0.0] += features_adjust
    maximum_entry_dbm = (np.max(data) / power_scale) - power_offset
    print(" (Max adjusted power = {0} dBm)".format(maximum_entry_dbm))
    
    #Checks if there is at least one point above the minimum adjusted threshold (-100dBm)
    if (maximum_entry_dbm <= - 100.0):
        print("No data entry above the minimum power cutoff (-100dBm)")
    
    #1D -> 2D
    features_2d = data.reshape(beamformings, time_slots)
    return(features_2d)
        
        
        
#Functions
#############################################





#Loads the dataset
print("Loading the dataset...")
with open(data_file, 'rb') as f:
  features, labels, invalid_slots = pickle.load(f)
  
features_size = features.shape[1]

if(len(invalid_slots) > 0):
    print("WARNING - this dataset has some features removed, for full data preprocess again!")

    
#randomly selects and index / indexes
images_to_process = 2
index = np.random.randint(features.shape[0], size=images_to_process)


#Plots stuff
plt.figure(1, figsize = [7,4])


for i in range(images_to_process):

    plt.subplot(1, images_to_process, i+1)

    image_2d = plot_a_sample(index[i], time_slots, beamformings)
    
    X = (labels[index[i]][0] * grid[0]) - shift[0]
    Y = (labels[index[i]][1] * grid[1]) - shift[1]
    
    title = 'X=' + str(X) + ' Y=' + str(Y)
    plt.title(title)
    plt.xlabel('Beamforming Index')
    plt.ylabel('Sample Number')

    cax = plt.imshow(np.transpose(image_2d), vmin = 0.0, vmax = 1.2)
    
    
cbar = plt.colorbar(cax, ticks=[0.0, 0.2, 0.7, 1.2])
cbar.ax.set_yticklabels(['no_data', '-150dBm', '-100dBm', '-50dBm'])
 
#plt.show() #PLOTS things
plt.savefig('samples.pdf', format='pdf')
