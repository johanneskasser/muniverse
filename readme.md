Minimal repository for EMG decomposition benchmark.

Datacuration - (1) simulated data, (2) experimental data
Two data sets - (1) simulated-dataset, (2) experimental-dataset; that each have one file
One algorithm - (1) barebones FastICA with some simple modifications
One metric - (1) number of accepted units

With the above, this repository sets up a full workflow for the use case where a user tests their algorithm on both datasets.

#### Setup guide

Install neuromotion, biomime first
- other files include ARMS_model, model.pth from Biomime, muap_examples (not required?), poses.csv (not required?)