# Data Preprocessing

*This part of the repository is in C++, due to legacy issues. I intend to change it to python somewhere in the future, to simplify the project.*

The contents of this folder were created to convert the variable-length raw output data from the Wireless InSite simulator
(1 data entry per received ray, variable number of rays per position) into fixed-size data (the PDP for each BF).

The shared dataset already contains a preprocessed dataset, with a sampling frequency of 20 MHz.
This is the only parameter that can't be changed later on. If a different sampling frequency is desired,
the instructions bellow must be followed.


## Running Sequence

- Download `CIR_32.zip` and unzip the `cir_\*.txt`;
- Edit the `SAMPLE_FREQ` in `general_includes.hpp` to the desired frequency;
- Edit `main.cpp` at line 111, with the dowloaded files location;
- Compile using the makefile and run. A new file should be present in the `\data_processed` folder (`final_table`).