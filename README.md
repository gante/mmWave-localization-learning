# Beamformed Fingerprint Learning

[Last major update: 31-Aug-2018]

With 5G millimeter wave wireless communications, the resulting radiation reflects on most visible
objects, creating rich multipath environments. The radiation is thus significantly shaped by the obstacles
it interacts with, carrying latent information regarding the relative positions of the transmitter, the
obstacles, and the mobile receiver.

In this GitHub repository, the creation of beamformed fingerprints is achieved
through a pre-established codebook of beamforming patterns transmitted by a base station. Making use
of the aforementioned hidden information, deep learning techniques are employed to
convert the received beamformed fingerprints into a mobile device’s position. Compared to recent low-power
A-GPS implementations, the simulations show that the proposed method achieves 60× higher
energy efficiency*, while keeping similar accuracy. The average errors of down to 3.3 meters are obtained
on realistic outdoor scenarios, containing mostly non-line-of-sight positions, making it a very competitive
and promising alternative for outdoor positioning.

The following image shows the simulated results for the average error per covered position. Given that the transmitter 
is the red triangle at the center of the image, and most of the solid yellow shapes are buildings (see the other image 
below), it is possible to confirm that being in a NLOS position is not a constraint for the proposed system.

<p align="center">
  <img src="images/error_vs_position.PNG" width="480"/>
</p>

For more information, refer to papers section of this README file. If you find any error, please contact me (joao.gante@tecnico.ulisboa.pt).

*Hopefully, the paper for this will be available soon :)



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

This repository is split in two parts (each with their own internal README):
- data_preprocessing (converts the raw data into an organized table)
- ml_training (the actual ML part)

*The data pre-processing sequence only needs to be executed if a different sampling rate is desired (default = 20MHz)*

### Data

You can contact me through email to obtain access to the dataset (joao.gante@tecnico.ulisboa.pt).

The data was generated using the [Wireless InSite ray-tracing simulator](https://www.remcom.com/wireless-insite-em-propagation-software/) and a [high precision open-source 3D map of New York](http://www1.nyc.gov/site/doitt/initiatives/3d-building.page), made available by the New York City Department of Information Technology & Telecommunications. The simulation consists of a 400 by 400 meters area, centered at the [Kaufman Management Center](https://goo.gl/maps/xrqvT9VS59K2).

<p align="center">
  <img src="images/propagation.PNG" width="400"/>
</p>


### Prerequisites

- C++ compiler *(if different sampling rates are desired)*
- Python 3.x
- Tensorflow


## Authors

* **João Gante**

### Papers

"Beamformed Fingerprint Learning for Accurate Millimeter Wave Positioning" --- https://arxiv.org/abs/1804.04112 (and VTC Fall 2018)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **Leonel Sousa** and **Gabriel Falcão**, my PhD supervisors
