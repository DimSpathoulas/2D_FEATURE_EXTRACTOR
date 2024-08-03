# 2d Feature Exctractor based on 3d detections
Part of my master thesis: **Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**
## Overview
This module extracts a feature vector for each 3D detected object. It projects each 3D detected object to its corresponding camera and, for all projected objects in the particular image, extracts a 1024 feature vector from the FPN of MASK R-CNN.

n
