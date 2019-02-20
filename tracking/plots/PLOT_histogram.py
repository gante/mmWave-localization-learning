######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt


N_BINS = 1000

#------------------------------------------------------------------------------
print("Loading the predictions...")

distance_output = []
# all_labels = []
# non_zeroes = []

print("(Static...)")
with open('../results/static_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break

print("(Pedestrian...)")
with open('../results/ped_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break

print("(Car...)")
with open('../results/car_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            # non_zeroes += non_z
        except EOFError:
            break
#------------------------------------------------------------------------------

len_predictions = len(distance_output)

print("({0} predictions loaded)".format(len_predictions))

avg_distance = np.mean(distance_output)
sorted_distance = np.sort(distance_output)
distance_95 = sorted_distance[int(0.95 * len_predictions)]
print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))


#--------------- RMSE
RMSE = np.sqrt(np.mean(np.square(distance_output)))

print("RMSE = {0} m".format(RMSE))

#--------------- CDF (It's a cumulative histogram, but that's almost the same :D)

CDF_bin = [0]*N_BINS
print('\n\nComputing the Cumulative Histogram...')

for i in range(N_BINS):
    current_percentile = i/N_BINS
    CDF_bin[i] = sorted_distance[int(len_predictions * current_percentile)]


fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.45, 0.2, 0.4, 0.4]
ax2 = fig.add_axes([left, bottom, width, height])

x = CDF_bin
y = np.linspace(0, 1, N_BINS)
ax1.plot(x,y)

ax1.set_ylim([0,1])
ax1.set_xlim([0, x[-1]])
ax1.set_ylabel("Cumulative Histogram", fontsize=12)
ax1.set_xlabel("Error (m)", fontsize=12)

#mark 95%
distance_95 = sorted_distance[int(0.95 * len_predictions)]
ax1.plot([0,distance_95], [0.95,0.95], color = 'tab:orange', linestyle = ':')
ax1.plot([distance_95,distance_95], [0.0,0.13], color = 'tab:orange', linestyle = ':')
ax1.plot([distance_95,distance_95], [0.19,0.95], color = 'tab:orange', linestyle = ':')

note1 = ax1.annotate("$95^{th}$"+" percentile:\n{:.3f} m".format(distance_95),
            xy=(distance_95, 0), xytext=(distance_95-2.5, 0.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )



#--------------- sub graph

ax2.plot(x[:900],y[:900])
ax2.set_xlim([0, x[900]])
ax2.set_ylim([0, 0.9])
ax2.set_title("Zoomed cumulative histogram,\n excluding the last 10%")

#mark 50%
distance_50 = sorted_distance[int(0.50 * len_predictions)]
ax2.plot([0,distance_50], [0.50,0.50], color = 'tab:orange', linestyle = ':')
ax2.plot([distance_50,distance_50], [0.0,0.50], color = 'tab:orange', linestyle = ':')

ax2.annotate("Median:\n{:.3f} m".format(distance_50), xy=(distance_50+0.1, 0), xytext=(distance_50+1, 0.15),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )

#plt.show()
plt.savefig('cdf.pdf', format='pdf')