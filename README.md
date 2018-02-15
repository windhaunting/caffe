# ACtion Tubelet detector

By Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid 

## Introduction

The ACtion Tubelet detector (ACT-detector) is a framework for action localization. 
It takes as input sequences of frames and outputs tubelets, i.e., sequences of bounding boxes with associated scores.

For more details, please refer to our [ICCV 2017 paper](https://hal.inria.fr/hal-01519812/document) and our [website](http://thoth.inrialpes.fr/src/ACTdetector/). 

JHMDB: frame and video mAP results.

method   |  frame-mAP | video-mAP |   |   |  |
:-------|:-----:|:-----:|:-----:|:-----:|:-----:|
| threshold | 0.5 | 0.2 | 0.5 | 0.75 | 0.5:0.95 |
[Wang16CVPR](https://arxiv.org/pdf/1604.07279.pdf) | 39.9 | - | 56.4 | - | - |
[Saha16BMVC](https://arxiv.org/pdf/1608.01529.pdf) | - |72.6 | 71.5 | 43.3 | 40.0 |
[Peng16ECCV w/0 MR](https://hal.inria.fr/hal-01349107v3/document) | 56.9 | 71.1 | 70.6 | 48.2 | 42.2 |
[Peng16ECCV with MR](https://hal.inria.fr/hal-01349107v3/document) | 58.5 |74.3 | 73.1 | - | - |
[Singh17ICCV](https://arxiv.org/pdf/1611.08563.pdf) | - | 73.8 | 72.0 | 44.5 | 41.6 | 
[Hou17ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_Tube_Convolutional_Neural_ICCV_2017_paper.pdf)  | 61.3 | **78.4** | **76.9** | - | - | 
**ACT-detector** | **65.7** | 74.2 | 73.7 | **52.1** | **44.8** |

UCF101: frame and video mAP results (with * we denote the UCF101v2 annotations from [here](https://github.com/gurkirt/corrected-UCF101-Annots)). For future experiments, please use the UCF101v2. 

method   |  frame-mAP | video-mAP |   |   |  |
:-------|:-----:|:-----:|:-----:|:-----:|:-----:|
| threshold | 0.5 | 0.2 | 0.5 | 0.75 | 0.5:0.95 |
[Saha16BMVC*](https://arxiv.org/pdf/1608.01529.pdf) | - | 66.7 | 35.9 | 7.9 | 14.4 |
[Peng16ECCV w/0 MR](https://hal.inria.fr/hal-01349107v3/document) | 64.8 | 71.8 | 35.9 | 1.6 | 8.8 |
[Peng16ECCV with MR](https://hal.inria.fr/hal-01349107v3/document) | 65.7 | 72.9 | - | - | - |
[Peng16ECCV with MR*](https://hal.inria.fr/hal-01349107v3/document) | - | 73.5 | 32.1 | 2.7 | 7.3 |
[Singh17ICCV*](https://arxiv.org/pdf/1611.08563.pdf) | - | 73.5 | 46.3 | 15.0 | 20.4 |
[Hou17ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_Tube_Convolutional_Neural_ICCV_2017_paper.pdf)  | 41.4 | 47.1 | - | - | - | 
**ACT-detector*** | **69.5** | **76.5** | **49.2** | **19.7** | **23.4** |
**ACT-detector** | **67.1** | **77.2** | **51.4** | **22.7** | **25.0** |

You can find the per-class frame-AP and video-AP results ([Tables 3 and 4 in our paper](https://hal.inria.fr/hal-01519812/document)) on UCF-Sports, JHMDB and UCF-101 [here](https://docs.google.com/spreadsheets/d/1WhOvsV-_ioel6vrpMskWayxahdMBUkXkQQzRlN3HrU0/edit?usp=sharing).

## Citing ACT-detector

If you find ACT-detector useful in your research, please cite: 

    @inproceedings{kalogeiton17iccv,
      TITLE = {Action Tubelet Detector for Spatio-Temporal Action Localization},
      AUTHOR = {Kalogeiton, Vicky and Weinzaepfel, Philippe and Ferrari, Vittorio and Schmid, Cordelia},
      YEAR = {2017},
      BOOKTITLE = {ICCV},
    }
    
## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Testing](#testing)
5. [Evaluation](#evaluation)
6. [Run on a new dataset](#newdataset)

## Installation

1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/vkalogeiton/caffe.git
  cd caffe
  git checkout act-detector
  ```
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

## Datasets

To download the ground truth tubes, run the script:

    ./cache/fetch_cached_data.sh ${dataset_name} # dataset_name: UCFSports, JHMDB, UCF101, UCF101v2

This will populate the `cache` folder with three `pkl` files, one for each dataset. 
For more details about the format of the `pkl` files, see `act-detector-scripts/Dataset.py`. 

If you want to reproduce exactly our results as reported in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document), 
we also provide the RGB and flow files for the three datasets we use. 

1. UCF-Sports

You can download the frames (1.5GB) and optical flow (42MB): 

    ./data/UCFSports/get_ucfsports_data.sh number # number = 0 for RGB Frames and 1 for optical flow

2. J-HMDB  

You can download the frames (4.2GB), optical flow (39MB) and ground truth annotations: 

    ./data/JHMDB/get_jhmdb_data.sh number # number = 0 for for RGB Frames and 1 for optical flow
    
3. UCF-101

You can download the frames (4.4GB), optical flow (860MB) and ground truth annotations: 

    ./data/UCF101/get_ucf101_data.sh number # number = 0 for for RGB Frames and 1 for optical flow

These will create the `Frames` and `FlowBrox04` folders in the directory of each dataset. 

Note that in `act-detector-scripts/Dataset.py` you need to update the `ROOT_DATASET_PATH` path
with your dataset path. For instance, if you the action localization datasets using the above scripts, you should update: `ROOT_DATASET_PATH=/CURRENT_CAFFE_PATH/data/dataset_name/`

You can find the UCF101v2 frames [here](https://github.com/gurkirt/corrected-UCF101-Annots).

## Training 

1. We provide the prototxt used for our experiments for UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in: `caffe/models/ACT-detector/${dataset_name}`.

2. Download the RGB and FLOW5 initialization models pre-trained on ILSVRC 2012: 

        ./models/ACT-detector/scripts/fetch_initial_models.sh
  
This will download the caffemodels:
`caffe/models/ACT-detector/initialization_VGG_ILSVRC16_K6_RGB.caffemodels` and 
`caffe/models/ACT-detector/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodels`

3. We provide an example of training commands for a `${dataset_name}`: 

i. RGB 
        
    export PYTHONPATH="$./act-detector-scripts:$PYTHONPATH"                       # path of act-detector 
    ./build/tools/caffe train \
    -solver models/ACT-detector/${dataset_name}/solver_RGB.prototxt \             # change dataset_name 
    -weights models/ACT-detector/initialization_VGG_ILSVRC16_K6_RGB.caffemodel \
    -gpu 0                                                                        # gpu id

ii. 5 stacked Flows

    export PYTHONPATH="$./act-detector-scripts:$PYTHONPATH"                       # path of act-detector 
    ./build/tools/caffe train \
    -solver models/ACT-detector/${dataset_name}/solver_FLOW5.prototxt \           # change dataset_name 
    -weights models/ACT-detector/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodel \
    -gpu 0                                                                        # gpu id

where `${dataset_name}` can be: `UCFSports`, `JHMDB`, `JHMDB2`, `JHMDB3`, `UCF101` or `UCF101v2`. 

## Testing

1. If you want to reproduce our results for the UCF-Sports, J-HMDB (3 splits) and UCF-101 datasets, you need to download our trained caffemodels. 
To obtain them for sequence length K=6, run from the main caffe directory for each dataset:

       ./models/ACT-detector/scripts/fetch_models.sh ${dataset_name} # change dataset_name 

This will download one `RGB.caffemodel` and one `FLOW5.caffemodel` for each dataset. 
These are stored in `models/ACT-detector/${dataset_name}`.

2. Next step is to extract tubelets. To do so, run: 

       python act-detector-scripts/ACT.py "extract_tubelets('${dataset_name}', gpu=-1)" # change dataset_name, -1 is for cpu, otherwise 0,...,n for your gpu id 
       
The tubelets are stored in the folder called `act-detector-results`. 
Note that the test is not efficient and can be coded more efficiently by extracting features once per frame. 

3. For creating tubes, you can run the following:

       python act-detector-scripts/ACT.py "BuildTubes('${dataset_name}')"     # change dataset_name 

The tubelets are stored in the folder called `results/ACT-detector`. 

For all cases `${dataset_name}` can be: `UCFSports`, `JHMDB`, `JHMDB2`, `JHMDB3`, `UCF101` or `UCF101v2`. 

## Evaluation 

1. For evaluating the per-frame detections, we provide scripts for frame-mAP, frame-MABO and frame-Classification. You can run them as follows: 
       
       python act-detector-scripts/ACT.py "frameAP('${dataset_name}')"       # change dataset_name 
       python act-detector-scripts/ACT.py "frameMABO('${dataset_name}')"
       python act-detector-scripts/ACT.py "frameCLASSIF('${dataset_name}')"
       
2. For evaluating the tubes, we provide scripts for video-mAP. You can run it as follows:

       python act-detector-scripts/ACT.py "videoAP('${dataset_name}')"       # change dataset_name 
       
## Run on a new dataset <a id="newdataset"></a>

If you want to run the ACT-detector on another dataset, you need the deploy, solver and train files. 
You can generate them as follows:

    python act-detector-scripts/ACT_create_prototxt.py ${dataset_name} False # change dataset_name, False if RGB, True if FLOW5

For all cases `${dataset_name}` can be: `UCFSports`, `JHMDB`, `JHMDB2`, `JHMDB3`, `UCF101` or `UCF101v2`. 

This will create a folder in `models/ACT-detector/` called `generated_${dataset_name}` containing the `deploy_${modality}.prototxt`, `train_${modality}.prototxt` and `solver_${modality}.prototxt`, where `${modality}` is `RGB` or `FLOW5`. 
Note that you need to modify the `ct-detector-scripts/Dataset.py` file to contain your dataset.        

## Models for sequence length K=8

You can download the RGB and FLOW5 initialization models pre-trained on ILSVRC 2012: 

        ./models/ACT-detector/scripts/fetch_initial_modelsK.sh 8 # K=8
  
This will download the caffemodels:
`caffe/models/ACT-detector/initialization_VGG_ILSVRC16_K8_RGB.caffemodels` and 
`caffe/models/ACT-detector/initialization_VGG_ILSVRC16_K8_FLOW5.caffemodels`
