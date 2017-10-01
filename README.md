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

## Citing

If you find ACT-detector useful in your research, please cite: 

    @inproceedings{kalogeiton17iccv:hal-01519812,
      TITLE = {{Action Tubelet Detector for Spatio-Temporal Action Localization}},
      AUTHOR = {Kalogeiton, Vicky and Weinzaepfel, Philippe and Ferrari, Vittorio and Schmid, Cordelia},
      YEAR = {2017},
      MONTH = Oct,
      BOOKTITLE = {{ICCV 2017 - IEEE International Conference on Computer Vision}},
      ADDRESS = {Venice, Italy},
    }
    
## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Data](#data)

## Installation -- FIX ME 

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


## Models

We provide the prototxt used for our experiments for UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in: `caffe/models/`. 

## Data

We also provide our trained models that are trained on different datasets: UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are the models used to produce our results in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document).

From the act-detector folder, run the model fetch script: `./data/scripts/fetch_models.sh`.

This will populate the `caffe/data` folder with the models. 
See `caffe/data/README.md` for details.

