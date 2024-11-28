# 2d Feature Exctractor based on 3d detections
Part of my master thesis: **Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**
## Overview
This module extracts a feature vector for each 3D detected object.

It projects each 3D detected object to its corresponding camera and, for all projected objects in the particular image, extracts a 1024 feature vector from the Feature Pyramid Network (FPN) of MASK R-CNN.


The final output of the module is a dictionary containing for each sample, the state vector of the object (in global frame), its point cloud features and its feature vector.

## TODO
1. Add instructions on the README on how to run the script.  
1.1 Which environment to create, activate it first and then run whatever is needed.  
1.2 What weights are needed, where to find them and where to place them.  
1.3 For me, I need to add `export PYTHONPATH=$PYTHONPATH:/mrcnn` before running anything.

export PYTHONPATH=$PYTHONPATH:/mrcnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin

python projection/feature_extractor_v2.py
python projection/main.py
2. Probably is (?): 
`python projection/main.py --version v1.0-trainval --data_root /second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval --detection_file /home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_train.npy --output_file mrcnn_train.pkl`  
or  
`python projection/main.py --version v1.0-trainval --data_root /second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval --detection_file /home/ktsiakas/thesis_new/PC_FEATURE_EXTRACTOR/tools/centerpoint_predictions_val.npy --output_file mrcnn_val.pkl`  

3. L400: `if i % 500 == 0:`. What happens for the last samples? For example, we go up to 1200. One update for the 500, one update for 1000, what happens to the last 200? Do we update them for sure? I guess yes, because ofthe L484, just to be sure. 

4. L268, remove the print sample token, just breaks the progress bar.  

5. Everytime we run the script, we write on the .pkl file. Does it reset after every run? Or it always appends data there? I mean, i run once to test something, and writes a couple of data. Then I run again, the file should be manually deleted before running?

3. [ANS] yes L484 ensures last 200 samples will be saved

5. [ANS] the prior data will be deleted on each run if the same output file is given. no need to delete it manually