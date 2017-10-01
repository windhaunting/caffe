# ACtion Tubelet detector

By Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid 

## Introduction

The ACtion Tubelet detector (ACT-detector) is a framework for action localization. 
It takes as input sequences of frames and outputs tubelets, i.e., sequences of bounding boxes with associated scores.

For more details, please refer to our [ICCV 2017 paper](https://hal.inria.fr/hal-01519812/document) and our [website](http://thoth.inrialpes.fr/src/ACTdetector/). 

Video mAP results on J-HMDB and UCF-101. 

method   |  J-HMDB |  |   |   | UCF-101 |   |    |   |
:-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| threshold | 0.2 | 0.5 | 0.75 | 0.5:0.95 | 0.2 | 0.5 | 0.75 | 0.5:0.95 |
[Peng16ECCV w/0 MR](https://hal.inria.fr/hal-01349107v3/document) | 71.1 | 70.6 | 48.2 | 42.2 | 71.8 | 35.9 | 1.6 | 8.8 |
[Peng16ECCV with MR](https://hal.inria.fr/hal-01349107v3/document) |**74.3** | 73.1 | - | - | 72.9 | - | - | - |
[Saha16BMVC](https://arxiv.org/pdf/1608.01529.pdf) |72.6 | 71.5 | 43.3 | 40.0 | 66.7 | 35.9 | 7.9 | 14.4 |
[Singh17ICCV](https://arxiv.org/pdf/1611.08563.pdf) | 73.8 | 72.0 | 44.5 | 41.6 | 73.5 | 46.3 | 15.0 | 20.4 |
**ACT-detector** | 74.2 | **73.7** | **52.1** | **44.8** | **77.2** | **51.4** | **22.7** | **25.0** |

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

## Installation

1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone git@github.com:vkalogeiton/caffe.git
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

    ./cache/fetch_cached_data.sh --dataset_name # dataset name: UCFSports, JHMDB, UCF101

This will populate the `cache` folder with three `pkl` files, one for each dataset. 
For more details about the format of the `pkl` files, see `act-detector-scripts/Dataset.py`. 

If you want to reproduce exactly our results as reported in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document), 
we also provide the RGB and flow files for the three datasets we use. 

1. UCF-Sports

You can download the frames (1.5GB), optical flow (42MB) and ground truth annotations: 

    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/UCFSports-Frames.tar.gz | tar xvz  # frames, 1.5GB
    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/UCFSports-OF.tar.gz     | tar xvz  # optical flow, 42MB

2. J-HMDB  

You can download the frames (4.2GB), optical flow (39MB) and ground truth annotations: 

    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/JHMDB-Frames.tar.gz  | tar xvz  # frames, 4.2GB
    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/JHMDB-OF.tar.gz      | tar xvz  # optical flow, 39MB 
    
3. UCF-101

You can download the frames (4.4GB), optical flow (860MB) and ground truth annotations: 

    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/UCF101-Frames.tar.gz    | tar xvz  # frames, 4.4GB
    curl http://pascal.inrialpes.fr/data2/kalogeit/al-datasets/UCF101-OF.tar.gz.tar.gz | tar xvz  # optical flow, 860MB

These will create a `datasets` folder in your current directory. 

## Training 

1. We provide the prototxt used for our experiments for UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in: `caffe/models/ACT-detector/${dataset_name}`.

2. Download the RGB and FLOW5 initialization models pre-trained on ILSVRC 2012: 

        ./data/scripts/fetch_initial_models.sh.
  
This will download the caffemodels:
`caffe/data/initialization_VGG_ILSVRC16_K6_RGB.caffemodels` and 
`caffe/data/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodels`

3. We provide an example of training commands for a `${dataset_name}`: 

i. RGB 
        
        export PYTHONPATH="$./act-detector-scripts:$PYTHONPATH"          # path of act-detector 
        ./caffe/build/tools/caffe train \
        -solver models/${dataset_name}/solver_RGB.prototxt \             # change dataset name 
        -weights models/initialization_VGG_ILSVRC16_K6_RGB.caffemodel \
        -gpu 0                                                           # gpu id

ii. 5 stacked Flows

        export PYTHONPATH="$./act-detector-scripts:$PYTHONPATH"          # path of act-detector 
        ./caffe/build/tools/caffe train \
        -solver models/${dataset_name}/solver_FLOW5.prototxt \           # change dataset name 
        -weights models/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodel \
        -gpu 0                                                           # gpu id


## Testing

1. We provide the prototxt used for our experiments for UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in: `caffe/models/`. 

2.  To obtain the caffemodels used for initialization of the RGB and FLOW5 networks, run from the main caffe directory: 

        cd caffe/
        ./data/scripts/fetch_initial_models.sh.
  
This will download the intial caffemodels:
`caffe/data/initialization_VGG_ILSVRC16_K6_RGB.caffemodels` and 
`caffe/data/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodels`

Note that these modes are K=6 parallel streams of the ILSVRC 2012 caffemodel. ## FIX ME 

3. We also provide our trained models that are trained on different datasets: UCF-Sports, J-HMDB (3 splits) and UCF-101. 
   To obtain our trained caffemodels for sequence length K=6, run from the main caffe directory:

       cd caffe/
       ./data/scripts/fetch_models.sh.

This will download one `RGB.caffemodel` and one `FLOW5.caffemodel` for each dataset: UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in `data/${dataset_name}`.

These are the models used to produce our results in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document).

