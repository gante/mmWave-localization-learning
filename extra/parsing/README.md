# Data Preprocessing

*This part of the repository is in C++, due to legacy issues. I intend to change it to python somewhere in the future, to simplify the project.*

The contents of this folder were created to convert the variable-length raw output data from the Wireless InSite simulator
(1 data entry per received ray, variable number of rays per position) into fixed-size data (the PDP for each BF, per receiver position).


## What is this parsing doing?

If you open the raw outputs from Wireless InSite, you'll see something like:
```
85 4
1 155.272 2.9115e-06 -197.45
2 52.8623 1.8076e-06 -205.537
3 -132.509 1.9267e-06 -212.885
4 59.7084 1.81392e-06 -217.744
86 4
1 -61.6042 2.91135e-06 -197.37
2 105.881 1.8082e-06 -205.538
3 14.7594 1.92537e-06 -213.009
4 -117.474 1.81451e-06 -217.765
87 3
1 11.2384 1.80882e-06 -205.537
2 84.6592 1.92404e-06 -213.131
3 -80.4554 1.81511e-06 -217.785
```

The file format is as follows:
1. There are two types of rows -- type 1, containing two integers, and type 2, containing an integer and three floating-point numbers;
2. Type 1 row values depict the antenna index (in this case, com 1 to 160801) and the number of simulated rays received by that antenna;
3. Each file in CIR_32 contains Type 1 rows sorted by antenna index;
4. When a type 1 row indicates N received simulated rays, it is immediately followed by N type 2 rows, each containing information for 1 received ray;
5. Type 2 row values depict the index of the received ray (from 1 to N) for the receiver in question, the phase of the signal in the ray (in degrees), the time from the start of the transmission (in seconds), and the ray's power (in dBm). Please keep in mind that all transmitter/receiver gain is applied after the ray-tracing simulations -- you can see them in the data processing scripts before training the ML model, in the code repository.

The parsing code in this folder converts this data into the PDP for each transmitted beamformning/receiver position combination, at a fixed sampling frequency. Because we have the phase of the received signal, constructive/destructive interference can be applied. The code has plenty of comments, please check it if you'd like to know the individual steps applied. To load this data, please check the loading script example in Python (which starts [here](https://github.com/gante/mmWave-localization-learning/blob/master/bff_positioning/data/data_preprocessing.py#L111))

The shared dataset file (`final_table`) already contains a preprocessed dataset, with a sampling frequency of 20 MHz.
This is the only parameter that can't be changed later on. If a different sampling frequency is desired, the instructions bellow must be followed.


## Running Sequence

- Download `CIR_32.zip` and unzip the `cir_\*.txt`;
- Edit the `SAMPLE_FREQ` in `general_includes.hpp` to the desired frequency;
- Edit `main.cpp` at line 111, with the dowloaded files location;
- Compile using the makefile and run. A new file should be present in the `\data_processed` folder (`final_table`).
