'''
Plots paths on the map. [The internal functions depend on 2-generate_paths.py]
'''

import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Constants to define the output plot
NUMBER_PATHS = 5
PED_MARKER = 'o'
PED_COLOR = 'red'
PED_MARKER_SIZE = 3
CAR_MARKER = 's'
CAR_COLOR = 'mediumseagreen'
CAR_MARKER_SIZE = 3


def valid_locations_to_matrix(valid_locations):
    '''
    Given the valid locations, returns a matrix with True/False for each
        possible position
    '''

    list_of_valid_locations = list(valid_locations.keys())
    max_x = 0
    max_y = 0
    for item in valid_locations.keys():
        if item[0] > max_x:
            max_x = item[0]
        if item[1] > max_y:
            max_y = item[1]
    valid_loc_matrix = np.full([max_x + 1, max_y + 1], False)
    for item in list_of_valid_locations:
        valid_loc_matrix[item[0], max_y - item[1]] = True   #<----- flips Y!!!

    return valid_loc_matrix


def get_path_positions(inverse_index_mapping, path):
    '''
    Given a ready path (a sequence of 1D indexes), checks the
        inverse_index_mapping lookup table for the true x and y values.
    '''

    x_sequence = []
    y_sequence = []
    for index in path:
        this_position = inverse_index_mapping[str(index)]
        x_sequence.append(this_position[0] - 200)
        y_sequence.append(this_position[1] - 200)

    return x_sequence, y_sequence



if __name__ == "__main__":
    start = time.time()

    print("Loading the paths...")
    with open('../tracking_data/paths_test', 'rb') as f:
        paths = pickle.load(f)

    #Gets the valid locations matrix
    valid_loc_matrix = valid_locations_to_matrix(paths['s'])
    inverse_index_mapping = {str(value): key for key, value in paths['s'].items()}

    #plots the basic map
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    cax = plt.imshow((-1)*np.transpose(valid_loc_matrix), vmin = -1.0,
        vmax = 0.0, extent=[-200,200,-200,200]), #cmap="jet_r")   # <colormap>_r -> reverses
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')

    #fetches and plots PEDESTRIAN paths
    for i in range(NUMBER_PATHS):
        rand_idx = np.random.choice(len(paths['p']))
        considered_path = paths['p'][rand_idx]
        x_sequence, y_sequence = get_path_positions(inverse_index_mapping, considered_path)
        for idx in range(len(x_sequence)):
            ax.plot(x_sequence[idx], y_sequence[idx], marker=PED_MARKER,
                markersize=PED_MARKER_SIZE, color=PED_COLOR)
            # ^ please notice the inverse order [yes, I need to fix this indexing :( ]

    #fetches and plots CAR paths
    for i in range(NUMBER_PATHS):
        rand_idx = np.random.choice(len(paths['c']))
        considered_path = paths['c'][rand_idx]
        x_sequence, y_sequence = get_path_positions(inverse_index_mapping, considered_path)
        for idx in range(len(x_sequence)):
            ax.plot(x_sequence[idx], y_sequence[idx], marker=CAR_MARKER,
                markersize=CAR_MARKER_SIZE, color=CAR_COLOR)

    legend_elements = [matplotlib.lines.Line2D([0], [0], color=PED_COLOR,
                            marker=PED_MARKER, linestyle='', label='Pedestrians'),
                       matplotlib.lines.Line2D([0], [0], color=CAR_COLOR,
                            marker=CAR_MARKER, linestyle='', label='Vehicles')]

    ax.legend(handles=legend_elements, loc='lower left')

    plt.savefig('paths_map.pdf', format='pdf')

    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))
