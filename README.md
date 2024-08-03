# 2d Feature Exctractor based on 3d detections
Part of my master thesis: Probabilistic 3d multi-modal multi-object tracking using machine learning and analytic collision risk calculation for autonomous navigation.
## This module extracts a feature vector for each 3d detected object.
Project each 3d detected object to it's corresponding camera.
For all projected objects in the particular image extract a 1024 feature vector from the FPN of MASK R-CNN.
