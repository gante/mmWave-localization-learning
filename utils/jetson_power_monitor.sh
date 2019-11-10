#!/bin/bash
# to measure average power consumed in 30sec with 1sec sampling interval

# defines duration of measurement, interval of measurement (both in seconds)
# and the measureable rails
outfile="$HOME/monitor_results.txt"
duration=30
interval=1
RAILS=("VDD_IN /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
 "VDD_SYS_GPU /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"
 "VDD_SYS_CPU /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input"
 "VDD_SYS_SOC /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input"
 "VDD_SYS_DDR /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power2_input"
"VDD_4V0_WIFI /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power2_input")

# sets the counters to 0
for ((i = 0; i < ${#RAILS[@]}; i++)); do
 read name[$i] node[$i] pwr_sum[$i] pwr_count[$i] <<<$(echo "${RAILS[$i]} 0 0")
done

# defines the end time, and measures the rails with the set interval
end_time=$(($(date '+%s') + duration))
while [ $(date '+%s') -le $end_time ]; do
 for ((i = 0; i < ${#RAILS[@]}; i++)); do
 pwr_sum[$i]=$((${pwr_sum[$i]} + $(cat ${node[$i]}))) &&
 pwr_count[$i]=$((${pwr_count[$i]} + 1))
 done
 sleep $interval
done

# prints the results to $outfile
echo "RAIL,POWER_AVG" >> $outfile
for ((i = 0; i < ${#RAILS[@]}; i++)); do
 pwr_avg=$((${pwr_sum[$i]} / ${pwr_count[$i]}))
 echo "${name[$i]},$pwr_avg" >> $outfile
done
