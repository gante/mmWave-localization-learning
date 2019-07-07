'''
data_preprocessing.py

-> Converts the pre-processed binary file into a numpy array
'''

import numpy as np
import pickle
import os
import struct
import matplotlib.pyplot as plt


def data_to_dataset(data, input_w_labels):
    '''
    Converts the raw (floating point) input data into features and labels
    '''

    num_positions = int(num_elements / input_w_labels)
    features = []
    labels = []

    for i in range(num_positions):

        if (i%int(num_positions/10)==0) or (i==0):
            print("Status: {0} out of {1} positions converted".format(i, num_positions))

        tmp_features = []
        tmp_labels = []

        for j in range(input_w_labels):
            item = data[(i * input_w_labels) + j]
            if j == 0:      # Position data - X
                item += x_shift
                item /= (grid_x)
                tmp_labels.append(item)
            elif j == 1:    # Position data - Y
                item += y_shift
                item /= (grid_y)
                tmp_labels.append(item)
            else:
#Important notes here:
#1) item == 0 -> there is no data here (there are no values > 0)
#2) The check for the minimum power threshold (e.g. -100 dBm) is performed
#   after the noise is added, not here.
#3) Nevertheless, to speed up the code, filters out values with very little
#   power. For the default simulation parameters, this filters samples with
#   less than -170 dBm. Since the default "minimum_power" is -125 dBm [check
#   simulation_parameters.py for the meaning of this variable], this means
#   we can reliably test (log-normal) noises with STD up to 15 dB [margin =
#   (-125) - -170 = 45 dB = 3*STD of 15 dB]
                if(item < 0 and item > -(power_offset)):
                    tmp_features.append((item+power_offset) * power_scale)
                else:
                    assert item <= 0.0
                    tmp_features.append(0.0)

        features.append(tmp_features)
        labels.append(tmp_labels)

    return features, labels


if __name__ == "__main__":
    print("\n------------------------------------------------------------")
    print("PLEASE DOUBLE-CHECK THE SIMULATION PARAMETERS!")
    print("------------------------------------------------------------\n")

    #Runs "simulation_parameters.py" and keeps its variables
    exec(open("simulation_parameters.py").read(), globals())

    time_slots = max_time * sample_freq
    input_size = time_slots * beamformings
    input_w_labels = int(input_size + 2)

    #Loads the binary dataset
    print("Loading the binary dataset...")
    with open(data_file, mode='rb') as file:
        data_binary = file.read()
        file.close()

    #Converts the binary dataset into float 32
    print("Converting the binary to float_32...")
    print("[this may take a couple of minutes and it will not print any progress]")
    binary_size = os.path.getsize(data_file)
    num_elements = int(binary_size/4)
    data = struct.unpack('f'*num_elements, data_binary)
    del data_binary

    #Converts the dataset into features/labels
    print("Converting the dataset into features/labels...")
    features, labels = data_to_dataset(data, input_w_labels)
    del data

    #Converting the features/labels into numpy arrays
    print("\nConverting the features/labels into numpy arrays...")
    features = np.array(features)
    labels = np.array(labels)

    #Printing the ranges of the features
    print("[label] x range:", labels[:,0].min(), labels[:,0].max())
    print("[label] y range:", labels[:,1].min(), labels[:,1].max())
    print("[features] power range:", features[:].min(), features[:].max())


    #Removing "weak" time_slots
    # !! this is dataset dependent! The removed columns had little to no data !!
    if slice_weak_TS:
        print("[REMOVING WEAK TS ON:] Removing the specific time slots... ")

        mask = np.ones(time_slots*beamformings, dtype=bool)

        ts_to_delete = [0]

        max_ts_index = time_slots-1
        ts_slice_index = slice_weak_TS_start_remove

        for i in range((max_ts_index-ts_slice_index) +1):
            ts_to_delete.append(ts_slice_index + i)

        print("Slots to remove:", ts_to_delete)

        for i in range(time_slots*beamformings):
            #DIM 1 = BF, DIM 2 = TS
            if (i % time_slots) in ts_to_delete:
                mask[i] = False

        #removes those slots from the data
        print("Before TS reduction: ", features.shape)
        features = features[:,mask]
        print("After TS reduction: ", features.shape)


    #Detecting invalid slots [i.e. deletes all TS/BF combinations with no data
    #   for this dataset]
    # ----> This is helpful when ***NOT*** using convolutional networks; With
    #   CNNs, this wrecks the 2D structure of the data.
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

    #Delecting invalid [x, y] positions
    # ----> invalid positions = positions with no data (i.e. only zeroes)
    print("Detecting the invalid (data-less) positions... ",end='',flush=True)
    mask = np.ones(features.shape[0], dtype=bool)
    removed_pos = 0

    for i in range(features.shape[0]):
        if sum(features[i,:]) == 0:
            mask[i] = False
            removed_pos += 1

    features = features[mask,:]
    labels = labels[mask,:]
    print("{0} data-less positions removed.".format(removed_pos))


    #Final data reports
    print("Usable positions = {0}".format(features.shape[0]))

    if detect_invalid_slots:
        print("AVG Sparsity = {0}".format(sum(non_zeros) / (features.shape[0]*time_slots*beamformings)))

    #Storing the result
    print("Storing the result ...")
    with open(preprocessed_file, 'wb') as f:
        pickle.dump([features, labels, invalid_slots], f)

    #Optional: plots the existing data points on a 2D image
    # to_plot = np.full([int(grid_x+1), int(grid_y+1)], 0.0)
    # for pos in range(labels.shape[0]):
        # x = int(labels[pos,0] * grid_x)
        # y = int((1.0 - labels[pos,1]) * grid_y) #flips y
        # to_plot[x, y] = 1.0
    # plt.imshow(np.transpose(to_plot))
    # plt.show()
