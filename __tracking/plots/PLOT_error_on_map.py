######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
print("Loading the predictions...")

distance_output = []
all_labels = []
# non_zeroes = []

print("(Static...)")
with open('../results/static_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break

print("(Pedestrian...)")
with open('../results/ped_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break

print("(Car...)")
with open('../results/car_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break
#------------------------------------------------------------------------------


len_predictions = len(distance_output)
all_labels = np.asarray(all_labels)

print("({0} predictions loaded)".format(len_predictions))
print("x range: {0} - {1}".format(all_labels[:,0].min(), all_labels[:,0].max()))
print("y range: {0} - {1}".format(all_labels[:,1].min(), all_labels[:,1].max()))

avg_distance = np.mean(distance_output)
sorted_distance = np.sort(distance_output)
distance_95 = sorted_distance[int(0.95 * len_predictions)]
print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))



#creates empty data structures   ->ROW (X) MAJOR i.e. row1 -> row2 -> row3
position_entries = [[] for i in range(401*401)]
position_error = [100] * (401*401)
position_error = np.asarray(position_error)

#for each prediction: gets the position -> gets the matrix index -> stores the error
print("\nSorting the predictions by position...")
for i in range(len_predictions):
    x = all_labels[i,0] * 400   #[0,1] -> [0,400]
    y = (1.0-all_labels[i,1]) * 400       #flips y

    matrix_index = int(x * 401 + y)

    position_entries[matrix_index].append(distance_output[i])

#gets each position's average   (if it has no entries, sets as default value)
print("\nAveraging the results by position...")
for i in range(401*401):
    if len(position_entries[i]) > 0:
        position_error[i] = np.mean(position_entries[i])



position_error_2D = position_error.reshape(401, 401)


#plots the thing
fig = plt.figure(1)
ax = fig.add_subplot(111)
cax = plt.imshow(np.transpose(position_error_2D), vmin = 0.0, vmax = 10, extent=[-200,200,-200,200]) #cmap="jet_r")   # <colormap>_r -> reverses
ax.set_ylabel('Y (m)')
ax.set_xlabel('X (m)')
ax.plot(0,3,'r^')

# cbar = plt.colorbar(cax, ticks=[0, 5, 10, 15, 20, 25, 30])
cbar = plt.colorbar(cax, ticks=[0, 2, 4, 6, 8, 10])
# cbar.ax.set_yticklabels(['0', '5', '10', '15', '20', '25', '30+'])
cbar.ax.set_yticklabels(['0', '2', '4', '6', '8', '10+'])
cbar.set_label('Average Error (m)', rotation=270)


plt.savefig('error_map.pdf', format='pdf')
# plt.show()