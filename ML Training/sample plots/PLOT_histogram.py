######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt


n_bins = 1000


print("Loading the predictions...", end = '')
with open('predictions.pkl', 'rb') as f:
    distance_output, all_labels, non_zeroes = pickle.load(f)
 
len_predictions = len(distance_output)

print("({0} predictions loaded)".format(len_predictions))
 
avg_distance = np.mean(distance_output)
sorted_distance = np.sort(distance_output)
distance_95 = sorted_distance[int(0.95 * len_predictions)]
print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))


#--------------- RMSE
RMSE = np.sqrt(np.mean(np.square(distance_output)))

print("RMSE = {0} m".format(RMSE))

#--------------- CDF    
    
CDF_bin = [0]*n_bins 
print('\n\nComputing the CDF...    <--- correction: cumulative histogram')

for i in range(n_bins):
    current_percentile = i/n_bins
    CDF_bin[i] = sorted_distance[int(len_predictions * current_percentile)]
    

fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.45, 0.2, 0.4, 0.4]
ax2 = fig.add_axes([left, bottom, width, height])
    
x = CDF_bin
y = np.linspace(0, 1, n_bins)
ax1.plot(x,y)

ax1.set_ylim([0,1])
ax1.set_xlim([0, x[-1]])
ax1.set_ylabel("Cumulative Histogram", fontsize=12)
ax1.set_xlabel("Error (m)", fontsize=12)

#mark 95%
distance_95 = sorted_distance[int(0.95 * len_predictions)]
ax1.plot([0,distance_95], [0.95,0.95], color = 'tab:orange', linestyle = ':')
ax1.plot([distance_95,distance_95], [0.0,0.09], color = 'tab:orange', linestyle = ':')
ax1.plot([distance_95,distance_95], [0.19,0.95], color = 'tab:orange', linestyle = ':')

note1 = ax1.annotate("$95^{th}$ percentile:\n"+str(distance_95), xy=(distance_95+1, 0), xytext=(distance_95-10, 0.1),
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

ax2.annotate("Median:\n"+str(distance_50), xy=(distance_50+0.1, 0), xytext=(distance_50+1, 0.15),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )

#plt.show()
plt.savefig('cdf.pdf', format='pdf')