import matplotlib.pyplot as plt


#--------------------------------------
#DATA
x = [0,2,6,10] #noise levels

# avg_error_binarized = [6.176, 9.332, 10.248, 12.112]
avg_error_binarized = [6.176, 9.332, 10.620, 12.112]
avg_error_normalized = [4, 8.088, 11.496, 13.634]

# error_95_binarized = [14.57, 25.53, 28.59, 35.31]
error_95_binarized = [14.57, 25.53, 31.867, 35.31]
error_95_normalized = [10, 20.359, 32.205, 40.368]
#--------------------------------------


fig, ax1 = plt.subplots()

ax1.plot(x, avg_error_binarized, 'o:', color='tab:blue')
ax1.plot(x, avg_error_normalized, '^-', color='tab:blue')

ax1.set_ylabel('Average Error (m)', fontsize=12, color='tab:blue')
ax1.set_xlabel('Noise $\sigma$ (dB)', fontsize=12)
ax1.tick_params('y', colors='tab:blue')

ax1.set_ylim([0,45])
ax1.set_xlim([0,10])


ax1.grid(True)
# plt.show()
# plt.text(x_msb[0], y_msb[0]+0.7, 'Continuous [REF1]', fontsize=10, horizontalalignment='center')

ax2 = ax1.twinx()
ax2.plot(x, error_95_binarized, 'o:', color='tab:orange')
ax2.plot(x, error_95_normalized, '^-', color='tab:orange')
ax2.set_ylabel('$95^{th}$ Percentile Error (m)', fontsize=12, color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
ax2.set_ylim([0,45])


ax1.plot([], [], color='black', linestyle = ':', marker = 'o', label='Binary Data')
ax1.plot([], [], color='black', linestyle = '-', marker = '^', label='Floating-Point Data')

ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))



plt.savefig('noise_vs_error.pdf', format='pdf')
