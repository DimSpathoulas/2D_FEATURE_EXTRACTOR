# 2d Feature Exctractor based on 3d detections
Part of my master thesis: **Manifold learning on geometric and appearance features from 3d and 2d detectors.**
## Overview
This module extracts a feature vector for each 3D detected object. A review paper in english is under construction.

## Instructions
### 1. Prerequisites 
You should have already followed the instructions from [my geometric appearance feature extractor repo](https://github.com/DimSpathoulas/PC_FEATURE_EXTRACTOR.git)

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

## TODO
1. Add instructions on the README on how to run the script.  
1.1 Which environment to create, activate it first and then run whatever is needed.  
1.2 What weights are needed, where to find them and where to place them.  
1.3 For me, I need to add `export PYTHONPATH=$PYTHONPATH:/mrcnn` before running anything.

export PYTHONPATH=$PYTHONPATH:/mrcnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin

python projection/feature_extractor_v2.py
python projection/feature_extractor.py
2. Probably is (?): 
`CUDA_VISIBLE_DEVICES=1 python projection/feature_extractor.py --version v1.0-trainval --data_root /second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval --detection_file /home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_train.npy --output_file mrcnn_train.pkl`  
or  
`python projection/feature_extractor.py --version v1.0-trainval --data_root /second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval --detection_file /home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_val.npy --output_file mrcnn_val.pkl`  

3. L400: `if i % 500 == 0:`. What happens for the last samples? For example, we go up to 1200. One update for the 500, one update for 1000, what happens to the last 200? Do we update them for sure? I guess yes, because ofthe L484, just to be sure. 

4. L268, remove the print sample token, just breaks the progress bar.  

5. Everytime we run the script, we write on the .pkl file. Does it reset after every run? Or it always appends data there? I mean, i run once to test something, and writes a couple of data. Then I run again, the file should be manually deleted before running?

3. [ANS] yes L484 ensures last 200 samples will be saved

5. [ANS] the prior data will be deleted on each run if the same output file is given. no need to delete it manually