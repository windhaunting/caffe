# ACtion Tubelet detector

By Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid 

## Example of training commands for UCF-Sports

1. RGB 

        cd caffe/                                                       # act-detector caffe directory
        export PYTHONPATH="$(pwd)/act-detector-scripts:$PYTHONPATH"     # path of act-detector 
        ./caffe/build/tools/caffe train \
        -solver models/UCFSports/solver_RGB.prototxt \
        -weights models/initialization_VGG_ILSVRC16_K6_RGB.caffemodel \
        -gpu 0                                                          # gpu id

2. 5 stacked Flows

        cd caffe/                                                       # act-detector caffe directory
        export PYTHONPATH="$(pwd)/act-detector-scripts:$PYTHONPATH"     # path of act-detector 
        ./caffe/build/tools/caffe train \
        -solver models/UCFSports/solver_FLOW5.prototxt \
        -weights models/initialization_VGG_ILSVRC16_K6_RGB.caffemodel \
        -gpu 0                                                          # gpu id


