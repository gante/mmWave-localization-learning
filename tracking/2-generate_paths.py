'''
generate_paths.py

-> Computes possible paths, given the simulated data and the movement type
'''

import os
import sys
import time
import pickle
import numpy as np


def load_preprocessed_data(preprocessed_file, removed_invalid_slots,
    predicted_input_size):
    '''
    Loads the preprocessed data and performs basic shape checks. The returned
    array contains ALL possible labels (i.e. it also includes labels that are
    only possible with unlikely noise functuations )
    '''

    print("Loading the dataset...", end='', flush=True)
    with open(preprocessed_file, 'rb') as f:
        features, labels, invalid_slots = pickle.load(f)
    print(" done! Valid positions:", labels.shape)

    #These assertions can catch differences between the set parameters and
    #   the stored data, which avoids weird bugs
    input_size = features.shape[1]
    assert features.shape[0] == labels.shape[0]
    if (removed_invalid_slots == False):
        assert predicted_input_size == input_size

    print("[label] x range:", labels[:,0].min(), labels[:,0].max())
    print("[label] y range:", labels[:,1].min(), labels[:,1].max())
    return labels


def get_possible_positions(labels, grid, scale, shift):
    '''
    Returns a dictionary with all possible possitions, and their index in the
        original dataset (a lookup table).

    DICTIONARY SHAPE: {(x,y) : index}

    EXPECTED INPUTS:
        - labels: 2D array whose 1st dimention is the index of the entry, and
            the 2nd is the physical dimention (0 = x, 1 = y)
        - grid: 1D list with the physical sizes for each dimention
        - scale: scalar integer with the physical distance between samples
            (therefore, a given dimention will have "grid/scale" elements along
            its axis)
        - shift: the offset for the starting point in each dimention
    '''

    if scale > 1:
        print("WARNING - the TCN train stage is not yet working for undersampled "\
            "data. The non-valid positions of 'features' and 'labels' should "\
            "be popped!")

    #Creates an "empty" boolean 2D array
    x_elements = int((grid[0] / scale) + 1)
    y_elements = int((grid[1] / scale) + 1)
    valid_locations = {}

    #Fills in the boolean array with the data from "labels"
    for i in range(labels.shape[0]):
        this_x = int(round(labels[i,0] * grid[0]))
        this_y = int(round(labels[i,1] * grid[1]))

        assert this_x >= 0
        assert this_y >= 0

        if this_x % scale != 0:
            continue
        if this_y % scale != 0:
            continue

        #there can be no repeated entries
        assert (this_x, this_y) not in valid_locations
        valid_locations[(this_x, this_y)] = i

    print("Number of valid locations: ", len(valid_locations.keys()))
    return valid_locations


def get_moving_paths(path_options, path_type, valid_locations, time_steps):
    '''
    Given the valid locations and the path options & type, generates as many
        moving paths as the number of valid_locations. Returns a list
        of dataset indexes for each path (to sample that path's received
        radiation, just use those indexes!)

    valid_locations - dictionary with format = {(x, y): dataset_index}
    '''

    pi = np.pi
    number_paths = len(valid_locations.keys())

    list_of_valid_locations = list(valid_locations.keys())
    max_x = 0
    max_y = 0
    for item in list_of_valid_locations:
        if item[0] > max_x:
            max_x = item[0]
        if item[1] > max_y:
            max_y = item[1]
    valid_loc_matrix = np.full([max_x + 1, max_y + 1], False)
    for item in list_of_valid_locations:
        valid_loc_matrix[item[0], item[1]] = True

    # Sets movement options
    avg_speed = path_options[path_type + '_avg_speed']
    max_speed = path_options[path_type + '_max_speed']
    max_speed_adjust = path_options[path_type + '_speed_adjust']
    #converts to radians
    max_direction_adjust = (path_options[path_type + '_direction_adjust'] / \
        360.0) * 2 * pi
    #probability of [no change; full stop; direction adjust; speed adjust]
    move_probabilities = path_options[path_type + '_move_proba']

    moving_paths = []
    for idx in range(number_paths):
        valid_path = False
        while not valid_path:
            #initializes the speed as 0, and its position as in one of the
            #   valid_locations
            this_speed_x = 0.0
            this_speed_y = 0.0
            this_x, this_y = list_of_valid_locations[\
                np.random.choice(len(list_of_valid_locations))]
            this_sequence_indexes = [valid_locations[(int(round(this_x)),
                int(round(this_y)))]]
            movement_type = None
            direction = None

            #loop until time_steps ends
            for ts in range(time_steps - 1):
                #-------------------------------------------------------------
                # Updates the pedestrian direction and speed
                total_speed = np.sqrt(np.square(this_speed_x) + np.square(this_speed_y))

                #after full stop, restarts in random direction with avg speed
                if total_speed == 0.0:
                    direction = np.random.uniform(0, 2*pi)
                    this_speed_x = avg_speed * np.cos(direction)
                    this_speed_y = avg_speed * np.sin(direction)
                #full stop
                elif movement_type == 1:
                    this_speed_x = 0.0
                    this_speed_y = 0.0
                #direction adjust
                elif movement_type == 2:
                    direction_adjust = np.random.uniform(-max_direction_adjust,
                        max_direction_adjust)
                    direction += direction_adjust
                    this_speed_x = total_speed * np.cos(direction)
                    this_speed_y = total_speed * np.sin(direction)
                #speed adjust
                elif movement_type == 3:
                    speed_adjust = np.random.uniform(-max_speed_adjust,
                        max_speed_adjust)
                    total_speed += speed_adjust
                    total_speed = np.clip(total_speed, 0.0, max_speed)
                    this_speed_x = avg_speed * np.cos(direction)
                    this_speed_y = avg_speed * np.sin(direction)
                #otherwise, must be a "no_change" movement type
                else:
                    assert movement_type == 0


                #-------------------------------------------------------------
                # Given the speed/direction, updates position
                #[positions in meters, speed in meters per second, 1 sample per second]
                this_x += this_speed_x
                this_y += this_speed_y

                #-------------------------------------------------------------
                # Checks if position is valid: if it is, stores it, otherwise
                #   restarts the path
                x_index = int(round(this_x))
                y_index = int(round(this_y))
                if x_index > max_x or x_index < 0:
                    break
                elif y_index > max_y or y_index < 0:
                    break
                elif not valid_loc_matrix[x_index, y_index]:
                    break
                else:
                    this_touple = (x_index, y_index)
                    this_sequence_indexes.append(valid_locations[this_touple])

                #-------------------------------------------------------------
                # Selects next movement type
                if len(this_sequence_indexes) == time_steps:
                    valid_path = True
                movement_type = np.random.choice(4, p=move_probabilities)

        assert len(this_sequence_indexes) == time_steps
        moving_paths.append(this_sequence_indexes)

        sys.stdout.write('Moving paths progress: {0} out of {1}\r'.format(idx + 1,
            number_paths))
        sys.stdout.flush()

    return moving_paths


def generate_paths(path_options, valid_locations, time_steps):
    '''
    Generates a dictionary of paths, given the options (also given as a
    dictionary)
    '''

    #Initializes the paths as a dictionary of empty lists, with a field per
    #   path type ('s' for static, 'p' for pedestrian, and 'c' for car)
    paths = {'s': [], 'p': [], 'c': []}

    #processes the static paths (just stores the valid locations)
    if path_options['s_paths']:
        print("Getting static paths...")
        paths['s'] = valid_locations

    #processes the pedestrian paths
    if path_options['p_paths']:
        print("Getting pedestrian paths...")
        paths['p'] = get_moving_paths(path_options, 'p', valid_locations,
            time_steps)

    #processes the car paths
    if path_options['c_paths']:
        print("Getting car paths...")
        paths['c'] = get_moving_paths(path_options, 'c', valid_locations,
            time_steps)

    return paths


if __name__ == "__main__":
    start = time.time()

    #Loads the tracking simulation parameters
    exec(open("simulation_parameters.py").read(), globals())

    #Loads the preprocessed data
    labels = load_preprocessed_data(preprocessed_file, removed_invalid_slots,
        predicted_input_size)

    #Gets the list of possible positions, given the preprocessed data
    print("\nGenerating list with valid locations...")
    valid_locations = get_possible_positions(labels, grid,
        int(spatial_undersampling), shift)
    del labels

    #Gets a dictionary with the desired paths
    print("\nGenerating the paths dictionary...")
    paths = generate_paths(path_options, valid_locations, time_steps)

    #If the os path for the desired data doesn't exist, creates it
    if not os.path.exists(tracking_folder):
        os.makedirs(tracking_folder)

    #Stores the dictionary of paths
    print("\nStoring the paths ...")
    with open(path_file, 'wb') as f:
        pickle.dump(paths, f)

    #After path generation, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))
