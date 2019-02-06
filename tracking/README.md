# Tracking [EXPERIMENTAL]

This folder contains an add-on to the "core" project, using RNNs to track users
when sequences of positions are available. To train the system, follow the
steps below

## Running Sequence

- Run steps 1, 2, 3, and 4 in the previous folder (to run step #k, execute the
script whose name starts with "k-"). This will train the non-tracking
regressor, which will be used as "baseline" here.
- Meanwhile, prepare the desired settings in "tracking_parameters.py";
- Generate the paths (i.e. sequence of TRUE positions) with step 5;
- Obtain the data triplet [noisy_features; baseline_prediction; true_label]
with step 6. Steps 5 and 6 are independent, so you can run them both at the
same time of you have enough computing resources;
- Use this data to train the RNN in step 7.