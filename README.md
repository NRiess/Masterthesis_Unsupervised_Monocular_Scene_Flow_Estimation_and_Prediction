# Unsupervised monouclar scene flow estimation and prediction
This repository contains the code of several experiments conducted by Nicolas Riess in the course of a [Masterthesis with the title "Unsupervised Monocular Scene Flow Estimation and Prediction"](https://drive.google.com/file/d/11pfFOjjE__mwDkPL0yvW13yFKLTzyJ7k/view?usp=share_link). The scene flow estimator and predictor presented here is based on the ![Scene flow estimator of Junhwa Hur and Stefan Roth](https://arxiv.org/abs/2004.04143).

Using two subseqent image of only one camera, a point cloud is constructed as can be seen seen in the following. 
<p align="center"><img src="demo/validation_image_60_x123-swin_different_perspectives.gif" width="500" height="250"/> </p>

The results of our model can be seen in the next GIF. The first two images of each sequence are the input of the neural network. A scene flow, which computes the displacement between the corresponding point clouds of the two subsequent camera frames, is estimated. It is visualized with pink lines.
Susbsequently a scene flow is predicted between the point cloud corresponding of the latest camera frame and a predicted future point cloud.                             

<p align="center">
  Two most recent frames and one future frame for each sequence:       
</p>
  
<p align="center">
  <img src=demo/validation_images.gif width="400" height="100"/> 
</p>

<p align="center">
   Point clouds of two most recent frames, corresponding estimated scene flow, future frame and corresponding predicted scene flow:
   
   <img src=demo/validation_images_point_clouds_scene_flow_with_prediction.gif width="1000" height="400"/> 
</p>

## Create environment

### Locally:
The code has been developed with Anaconda (Python 3.7.12), PyTorch 1.10.0 and CUDA 
11.4 (Different Pytorch + CUDA version is also compatible).  
Please run the following commands:

  ```Shell
conda create -n sf-swin python=3.7
conda activate sf-win

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.4 -c pytorch
pip install tensorboard
pip install pypng==0.0.21
pip install colorama==0.4.4
pip install scikit-image==0.19.2
pip install pytz==2022.1
pip install tqdm==4.30.0
pip install future==0.18.2
  ```

Please go to the directory of this project and run the following command:
  ```Shell
cd ./models/correlation_package
python setup.py install
cd ../forwardwarp_package
python setup.py install
cd ../..
  ```
Alternatively, 
  ```Shell
bash install_modules.sh
  ```
can be run in the project directory.

Finally, 
  ```Shell
pip install timm==0.5.4
pip install pyyaml==6.0
pip install yacs==0.1.8
pip install matplotlib==3.5.2
pip install open3d==0.15.2
pip install protobuf==3.19.4
pip install azureml==0.2.7
pip install azureml.core==1.42.0
  ```
can be executed to install the remaining packages.
  
For PyTorch version > 1.3:
Please put the **`align_corners=True`** flag in the `grid_sample` function in the following files:
  ```
  augmentations.py
  losses.py
  models/modules_sceneflow.py
  utils/sceneflow_util.py
  ```

### On azure:
* Create an account and login in [azure](https://ml.azure.com) 
* Click on "Create"
* Choose the name "sf-swin"
* Select the environment type "Create a new docker context"
* Choose the file "Dockerfile"
* Build the environment

## Download dataset

Please download the following to datasets for the experiment:
  - [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) (synced+rectified data, please refer [MonoDepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for downloading all data more easily)
  - [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

To save space, we also convert the *KITTI Raw* **png** images to **jpeg**, following the convention from [MonoDepth](https://github.com/mrharicot/monodepth):
  ```
  find (data_folder)/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
  ```   
We also converted images in *KITTI Scene Flow 2015* as well. Please convert the png images in `image_2` and `image_3` into jpg and save them into the seperate folder **`image_2_jpg`** and **`image_3_jpg`**.  

To save space further, the Velodyne point data in KITTI raw data can be deleted and optionally the [*Eigen Split Projected Depth*](https://drive.google.com/file/d/1a97lgOgrChkLxi_nvRpmbsKspveQ6EyD/view?usp=sharing) for the monocular depth evaluation on the Eigen Split can be downloaded. We converted the velodyne point data of the Eigen Test images in the numpy array format using code from [MonoDepth](https://github.com/mrharicot/monodepth). After downloading and unzipping it, it can be merged with the KITTI raw data folder.  

### On azure:
Please execute the following steps after downloading the two datasets:
* Create a new folder locally named "kitti"
* Paste the KITTI Raw Data and the KITTI Scene Flow 2015 dataset into this folder
* Login in [azure](https://ml.azure.com) 
* Click on "Data"
* Click on "Create" 
* Choose option "From local files"
* Choose the name "kitti"
* Select "file"
* Click on "Browse", "Browse folder" and select folder "kitti"
* Click on "Next" and confirm
* Create new datastore and name it "deepdatasets" or choose existing datastore and change the name in "run_with_azure.py" accordingly


## Start training

### Locally:
Please run main.py using the arguments --debug=False and --azure=False. Additonally, one of of the follow strings for the argument --version can be chosen:
* "correlation": Baseline that computes the correlation
* "concatenated_inputs": The feature pyramid are concatenated to learn temporal relations already in the encoder
* "stacked": Swin transformer replaces the correlation operation and receives the difference of the features and warped features as input
* "diff": Swin transformer replaces the correlation operation and receives the difference of the features and warped features as input
* "predict": Monocular Scene Flow estimator and predictor that processes three subsequent frames as inputs

### On azure:
* Adjust the version number of the environment in "run_with_azure.py" according to the desired version 
* Type in the your subscription_id, resource_group and workspace_name in "run_with_azure.py" 
* Run the file to start the training.

## Evaluation
The individual models can be evaluated by setting --checkpoint=$PATH_TO_CHECKPOIN$ and --debug=False when running main.py.
