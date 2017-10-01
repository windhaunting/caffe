# ACtion Tubelet detector

By Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid 

## Data

We provide our trained models that are trained on different datasets (UCF-Sports, J-HMDB and UCF-101). 
These are the models used to produce our results in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document).

1.  To obtain the caffemodels used for initialization of the RGB and FLOW5 networks, run from the main caffe directory: 

        cd caffe/
        ./data/scripts/fetch_initial_models.sh.
  
This will download the intial caffemodels:
`caffe/data/initialization_VGG_ILSVRC16_K6_RGB.caffemodels` and 
`caffe/data/initialization_VGG_ILSVRC16_K6_FLOW5.caffemodels`

Note that these modes are K=6 parallel streams of the ILSVRC 2012 caffemodel. ## FIX ME 

2. To obtain our trained caffemodels for sequence length K=6, run from the main caffe directory:

       cd caffe/
       ./data/scripts/fetch_models.sh.

This will download one `RGB.caffemodel` and onr `FLOW5.caffemodel` for each dataset: UCF-Sports, J-HMDB (3 splits) and UCF-101. 
These are stored in `data/${dataset_name}`.

These are the models used to produce our results in [Tables 2 and 3](https://hal.inria.fr/hal-01519812/document).
