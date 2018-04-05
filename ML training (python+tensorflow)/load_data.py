



##########################################################
#Data Loading

def create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler = None):

        
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
    noisy_labels = labels[mask,:]
    
    #doublecheck
    print(noisy_features.shape, noisy_labels.shape)
    
    #Applies the preprocessing, if needed
    if scaler is not None:
        noisy_features = scaler.fit_transform(noisy_features)
    
    return([noisy_features, noisy_labels])
    
            


#Loads the dataset, stored as 3 numpy arrays   [plus a couple of doublechecks]
print("Loading the dataset...", end='', flush=True)
with open(data_file, 'rb') as f:
    features, labels, invalid_slots = pickle.load(f)
    
input_size = features.shape[1]
print(" done! Features shape:", features.shape)


assert features.shape[0] == labels.shape[0]

if (removed_invalid_slots == False):
    assert predicted_input_size == input_size
  
  
#DATA PREPROCESSING [normalize]:
if binary_scaler:
    from sklearn.preprocessing import Binarizer
    scaler = Binarizer(0.1, copy=False)
    scaler_name = 'binarized'
else:
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer(copy=False)
    scaler_name = 'normalized'
  
  
#Creating the artificial test set
print("Creating the artificial test set...")
features_test, labels_test = create_noisy_features(features, labels, noise_std_converted, min_pow_cutoff, scaler)
    
#Data Loading
##########################################################