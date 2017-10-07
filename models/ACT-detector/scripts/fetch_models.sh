#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

DATASET_NAME=$1
echo "Downloading trained ACT-detector caffemodels for $DATASET_NAME.."

FILE=act-detector-caffemodels-$DATASET_NAME.tgz

wget http://pascal.inrialpes.fr/data2/act-detector/downloads/trained_models/$FILE

echo "Unzipping..."

tar zxvf $FILE -C models/ACT-detector/$DATASET_NAME/ && rm -f $FILE

echo "Done."
