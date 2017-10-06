#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

DATASET_NAME=$1

declare -a ALL_DATASET_NAMES=("UCFSports" "JHMDB" "UCF-101")

if [[ $ALL_DATASET_NAMES =~ (^|[[:space:]])$DATASET_NAME($|[[:space:]]) ]] ; then
  echo "Downloading $DATASET_NAME ground truth data"
else
  echo "Dataset name does not exist. Please select a valid dataset."
  exit 0 
fi

wget http://pascal.inrialpes.fr/data2/act-detector/downloads/cache/$DATASET_NAME-GT.pkl

echo "Done."