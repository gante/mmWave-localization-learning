# [MAJOR UPDATE IN PROGRESS, PLEASE CHECK BACK IN A FEW DAYS]

# mmWave Localization Learning

With millimeter wave (mmWave) wireless communications, the resulting radiation reflects on most visible objects, creating
rich multipath environments, namely in urban scenarios. The radiation captured by a listening device is thus shaped by the
obstacles encountered, which carry latent information regarding their relative positions. 

In this repository, a system to convert the received mmWave radiation into the device’s position is proposed, making use
of the aforementioned hidden information. Using deep learning techniques and a pre-established codebook of beamforming
patterns transmitted by a base station, my simulations show that average estimation errors below 10 meters are achievable in
realistic outdoors scenarios that contain mostly non-line-of-sight positions, paving the way for new positioning systems. 

For more information, refer to papers section of this readme file.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Data

You can contact me through email to obtain the dataset (joaofranciscocardosogante@gmail.com). 

The data was generated using the [Wireless InSite ray-tracing simulator](https://www.remcom.com/wireless-insite-em-propagation-software/) and a [high precision open-source 3D map of New York](http://www1.nyc.gov/site/doitt/initiatives/3d-building.page), made available by the New York City Department of Information Technology & Telecommunications. The simulation consists of a 400 by 400 meters area, centered at the [Kaufman Management Center](https://goo.gl/maps/xrqvT9VS59K2).


### Prerequisites

- C++ compiler (if different sampling rates are desired)
- Python 3.x
- Tensorflow


## Running Sequence

The data pre-processing sequence only needs to be executed if a different sampling rate is desired (default = 20MHz)

### Data Pre-Processing
- Download the 'cir_\*.txt' files; 
- Edit the SAMPLE_FREQ at 'general_includes.hpp' to the desired frequency; 
- Edit 'main.cpp' at line 111, with the dowloaded files location; 
- Compile using the makefile and run. A new file should be present in the '\data_processed folder' ('final_table').

### Positioning Learning
- Download/copy the 'final_table' file into the '\mmWave-localization-learning\ML training (python+tensorflow)' folder; 
- Run 'data_preprocessing.py'; 
- Edit 'simulation_parameters.py' to the desired settings; 
- Run 'DNN_train.py' to train the NN; 
- Run 'DNN_predict.py' (inside the '\plots and data analysis' folder) to sample predictions;
- Visualize the data using the provided scripts.
 

## Authors

* **João Gante**

### Papers

"Beamformed Fingerprint Learning for Accurate Millimeter Wave Positioning" --- https://arxiv.org/abs/1804.04112

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Leonel Sousa and Gabriel Falcão, my PhD supervisors
