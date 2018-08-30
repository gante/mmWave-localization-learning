import math
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer

#Runs "simulation_parameters.py" and keeps its variables    [simulation parameters]
exec(open("simulation_parameters.py").read(), globals())

##########################################################
#Data Loading


def create_noisy_features(features, labels, noise_std_converted, 
                          min_pow_cutoff, scaler = None, only_16_bf = False, 
                          undersampling = 1):

    #undersamples the BFs, if wanted
    if only_16_bf:
        mask = np.ones(time_slots*beamformings, dtype=bool)
        for i in range(time_slots*beamformings):
            #DIM 1 = BF, DIM 2 = TS
            if (i//time_slots)%2 == 0:
                mask[i] = False
        features = features[:,mask]
        print("[warning: using 16 bfs]", features.shape)
    
    #undersamples spacially, if wanted
    if undersampling > 1:
        undersampling = int(undersampling) #just in case
        
        mask = np.ones(labels.shape[0], dtype=bool)
        for i in range(labels.shape[0]):
            label_x_scaled = int(labels[i,0] * 400)
            if label_x_scaled % undersampling > 0:
                mask[i] = False
            else:
                label_y_scaled = int(labels[i,1] * 400)
                if label_y_scaled % undersampling > 0:
                    mask[i] = False

        features = features[mask,:]
        labels = labels[mask,:]
        
        print("[warning: undersampling by {0}]".format(undersampling))
        print(features.shape)
    
    
    #the features here should be in range [0, ~1.2]
    noise = np.random.normal(scale = noise_std_converted, size = features.shape)
    noisy_features = features + noise
    noisy_features[noisy_features < min_pow_cutoff] = 0
    
      
    #removes the entries containing only 0
    mask = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        
        this_samples_sum = np.sum(noisy_features[i,:])
        if this_samples_sum < 0.01:
            mask[i] = False
            
    noisy_features = noisy_features[mask,:]
    noisy_labels = labels[mask,:]

    #doublecheck
    assert noisy_features.shape[0] == noisy_labels.shape[0]
    assert noisy_labels.shape[1] == 2
    assert noisy_features.shape[1] == features.shape[1]

    #Applies the preprocessing, if needed
    if scaler is not None:
        noisy_features = scaler.fit_transform(noisy_features)
    return([noisy_features, noisy_labels])
    
    
    
    
def position_to_class(labels, lateral_partition):

    class_indexes = []
    n_classes = lateral_partition ** 2
    
    for i in range(labels.shape[0]):
        
        x_index = int(math.floor(labels[i,0] * lateral_partition))
        if(x_index == lateral_partition): x_index = lateral_partition-1
        
        y_index = int(math.floor(labels[i,1] * lateral_partition))
        if(y_index == lateral_partition): y_index = lateral_partition-1
        
        true_index = (y_index * lateral_partition) + x_index
        
        class_indexes.append(true_index)
    
    class_indexes = np.asarray(class_indexes)
    
    return(class_indexes)
    
            


#Loads the dataset, stored as 3 numpy arrays   [plus a couple of doublechecks]
print("Loading the dataset...", end='', flush=True)
with open(preprocessed_file, 'rb') as f:
    features, labels, invalid_slots = pickle.load(f)
    
input_size = features.shape[1]
print(" done! Features shape:", features.shape)


assert features.shape[0] == labels.shape[0]

if (removed_invalid_slots == False):
    assert predicted_input_size == input_size
  
  
#DATA PREPROCESSING [binarize/normalize]:
if binary_scaler:  
    scaler = Binarizer(0.1, copy=False)
    scaler_name = 'binarized'
else:    
    scaler = Normalizer(copy=False)
    scaler_name = 'normalized'
  
  
    
#Data Loading
##########################################################