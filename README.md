# self-mono-sf-simple

## Create environment

### Locally:
The code has been developed with Anaconda (Python 3.7.12), PyTorch 1.10.0 and CUDA 
11.4 (Different Pytorch + CUDA version is also compatible).  
Please run the following commands:

  ```Shell
  conda env create -f environment.yml
  conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.4 -c pytorch
  conda activate dfn-pytorch
  ./install_modules.sh
  ```
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

## Download dataset:

Please download the following to datasets for the experiment:
  - [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) (synced+rectified data, please refer [MonoDepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for downloading all data more easily)
  - [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

To save space, we also convert the *KITTI Raw* **png** images to **jpeg**, following the convention from [MonoDepth](https://github.com/mrharicot/monodepth):
  ```
  find (data_folder)/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
  ```   
We also converted images in *KITTI Scene Flow 2015* as well. Please convert the png images in `image_2` and `image_3` into jpg and save them into the seperate folder **`image_2_jpg`** and **`image_3_jpg`**.  

To save space further, you can delete the velodyne point data in KITTI raw data and optionally download the [*Eigen Split Projected Depth*](https://drive.google.com/file/d/1a97lgOgrChkLxi_nvRpmbsKspveQ6EyD/view?usp=sharing) for the monocular depth evaluation on the Eigen Split. We converted the velodyne point data of the Eigen Test images in the numpy array format using code from [MonoDepth](https://github.com/mrharicot/monodepth). After downloading and unzipping it, you can merge with the KITTI raw data folder.  
  - [Eigen Split Projected Depth](https://drive.google.com/file/d/1a97lgOgrChkLxi_nvRpmbsKspveQ6EyD/view?usp=sharing)

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

### On azure:
* Adjust the version number of the environment in "run_with_azure.py" according to the version you want to use
* Type in the your subscription_id, resource_group and workspace_name in "run_with_azure.py" 
* Run the file to start the training.