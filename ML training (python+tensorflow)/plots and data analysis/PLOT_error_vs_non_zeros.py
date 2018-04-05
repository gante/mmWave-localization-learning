######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt



print("Loading the predictions...", end = '')
with open('predictions.pkl', 'rb') as f:
    distance_output, all_labels, non_zeroes = pickle.load(f)
 
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
    
x = list(range(1,n_bins+1))
bin_counts = np.asarray(bin_counts)
    
plt.subplot(211) 
plt.plot(x,bin_counts/1000)
plt.xlim([0,n_bins])
plt.ylim([0,np.max(bin_counts/1000)])
plt.ylabel("Sample Count (Thousands)")
plt.grid(True)
    
    
plt.subplot(212)    
plt.plot(x, non_zero_bins_mean, label='Average Error')
plt.fill_between(x, non_zero_bins_bot, non_zero_bins_top, facecolor='tab:blue', alpha=0.3, label = '$0-95^{th}$ Percentile Error')
plt.legend()
plt.xlim([0,n_bins])
plt.ylim([0,min(np.max(non_zero_bins_top), 50)])
plt.ylabel("Error (m)")
plt.xlabel("Number of Detected (Non-zero) Entries")
plt.grid(True)

plt.savefig('error_vs_nonzero.pdf', format='pdf')
# plt.show()