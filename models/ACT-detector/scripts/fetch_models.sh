#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR


DATASET_NAME=$1

declare -a ALL_DATASET_NAMES=("UCFSports" "JHMDB" "JHMDB2" "JHMDB3" "UCF-101")
if [[ $ALL_DATASET_NAMES =~ (^|[[:space:]])$DATASET_NAME($|[[:space:]]) ]] ; then
  echo "Downloading trained ACT-detector caffemodels for $DATASET_NAME.."
else
  echo "Dataset name does not exist. Please select a valid dataset."
  exit 0 
fi

FILE=act-detector-caffemodels-$DATASET_NAME.tgz

wget http://pascal.inrialpes.fr/data2/act-detector/downloads/trained_models/$FILE

echo "Unzipping..."

tar zxvf $FILE -C ../$DATASET_NAME/

echo "Done."