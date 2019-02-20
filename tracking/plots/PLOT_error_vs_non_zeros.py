######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt


MEMORY = 7 #there are at least these non-zeros per sample


#------------------------------------------------------------------------------
print("Loading the predictions...")

distance_output = []
# all_labels = []
non_zeroes = []

print("(Static...)")
with open('../results/static_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            non_zeroes += non_z.tolist()
        except EOFError:
            break

print("(Pedestrian...)")
with open('../results/ped_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            non_zeroes += non_z.tolist()
        except EOFError:
            break

print("(Car...)")
with open('../results/car_results.pkl', 'rb') as f:
    while 1:
        try:
            distance, labels, non_z = pickle.load(f)
            distance_output += distance
            # all_labels += labels
            non_zeroes += non_z.tolist()
        except EOFError:
            break
#------------------------------------------------------------------------------


len_predictions = len(distance_output)

print("({0} predictions loaded)".format(len_predictions))

avg_distance = np.mean(distance_output)
sorted_distance = np.sort(distance_output)
distance_95 = sorted_distance[int(0.95 * len_predictions)]
print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))


n_bins = int(np.max(non_zeroes))

non_zero_bins = [[] for i in range(n_bins)]
bin_counts = [0] * n_bins
print('\n\nGetting the bins...')

for i in range(len_predictions):
    this_bin = int(non_zeroes[i]) - 1   #-1 because bin number 0 corresponds to 1 non_zero entry
    non_zero_bins[this_bin].append(distance_output[i])
    bin_counts[this_bin] += 1


#Averaging the results
non_zero_bins_mean = [[] for i in range(n_bins)]
for i in range(n_bins):
    if len(non_zero_bins[i]) > 0:
        non_zero_bins_mean[i] = np.mean(non_zero_bins[i])
    else:
        non_zero_bins_mean[i] = non_zero_bins_mean[i-1]

#2.5% & 97.5% (i.e. 95% of the results)
non_zero_bins_bot = [[] for i in range(n_bins)]
non_zero_bins_top = [[] for i in range(n_bins)]
for i in range(n_bins):
    if len(non_zero_bins[i]) > 0:
        #sorts the list -> get top & bot values
        sorted_bin = np.sort(non_zero_bins[i])
        non_zero_bins_bot[i] = sorted_bin[0]
        non_zero_bins_top[i] = sorted_bin[int(bin_counts[i]*0.95)]
    else:
        non_zero_bins_bot[i] = non_zero_bins_bot[i-1]
        non_zero_bins_top[i] = non_zero_bins_top[i-1]

x = list(range(n_bins))
bin_counts = np.asarray(bin_counts)
assert len(x) == len(bin_counts) == len(non_zero_bins_mean), "{} vs {} vs {}".\
    format(len(x), len(bin_counts), len(non_zero_bins_mean))

#crops the empty values
cropped_entries = 0
while not non_zero_bins[cropped_entries]:
    cropped_entries += 1
assert cropped_entries >= (MEMORY - 1)
for i in range(cropped_entries):
    bin_counts[i] = 0
    non_zero_bins_bot[i] = non_zero_bins_bot[cropped_entries]
    non_zero_bins_top[i] = non_zero_bins_top[cropped_entries]
    non_zero_bins_mean[i] = non_zero_bins_mean[cropped_entries]

ax1 = plt.subplot(211)
ax1.plot(x,bin_counts, color='tab:orange')
ax1.set_xlim([0,n_bins])
ax1.set_ylim([0,np.max(bin_counts)])
# ax1.set_ylim([0,4000])
ax1.set_ylabel("Sequence Count")
# ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.grid(True)


plt.subplot(212)
plt.plot(x, non_zero_bins_mean, label='Average Error')
plt.fill_between(x, non_zero_bins_bot, non_zero_bins_top, facecolor='tab:blue', alpha=0.3, label = '$95\%$ Interval')
plt.legend()
plt.xlim([0,n_bins])
# plt.ylim([0,min(np.max(non_zero_bins_top), 50)])
plt.ylim([0,min(np.max(non_zero_bins_top), 20)])
plt.ylabel("Error (m)")
plt.xlabel("Number of Detected (Non-zero) Paths")
plt.grid(True)

plt.savefig('error_vs_nonzero.pdf', format='pdf')
# plt.show()