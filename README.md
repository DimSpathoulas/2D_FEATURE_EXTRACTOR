# 2d Feature Exctractor based on 3d detections
Part of my master thesis: **3D Multi-Modal Multi-Object Tracking via Machine Learning and Analytic Collision Risk Calculation for Autonomous Vehicles Navigation.**
## Overview
This module extracts a feature vector for each 3D detected object. A review paper in english is under construction.

## Instructions
### 1. Prerequisites 
You should have already followed the instructions from my geometric feature [extractor](https://github.com/DimSpathoulas/PC_FEATURE_EXTRACTOR.git).

### 2. Clone our repo and setup the environment
**This Mask R-CNN repo works with 2.1.0 tensorflow version.**
```bash
cd path_to_your_projects/2D_FEATURE_EXTRACTOR
conda create --name appearance_extractor python=3.7.16
git clone https://github.com/DimSpathoulas/2D_FEATURE_EXTRACTOR.git
conda env create -f environment.yml
conda activate appearance_extractor
```

### 3. Extract Appearance Features
Download the ```mask_rcnn_coco.h5``` pre-trained model from the [official repo](https://github.com/matterport/Mask_RCNN).
From 2D_FEATURE_EXTRACTOR run:
```bash
export PYTHONPATH=$PYTHONPATH:/mrcnn
python projection/feature_extractor.py --version v1.0-trainval --data_root /second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval --detection_file /home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_train.npy --output_file mrcnn_train.pkl
```
Change paths accordingly.
```v2.py``` is under construction.

## Acknowledgments
Built on top of [Mask_RCNN](https://github.com/matterport/Mask_RCNN).
