# PoseFromPointClouds
Tensorflow and Pytorch implementations for pose predictions from point clouds. The Tensorflow implementation is a simplified version of the one used for producing the results for "Cutting Pose Prediction from Point Clouds". The pytorch implementation is used for ongoing work on GNNs.

# Usage
1. edit train.py to select between datasets and network_type (optional)
2. python3 train.py
3. python3 test.py
4. python3 evaluate.py

# TODO
1. write more instructions
2. Add visualization scripts

# Context
The models are part of the systems shown here.
![](/figs/training_system.png)
![](/figs/inference_system.png)
