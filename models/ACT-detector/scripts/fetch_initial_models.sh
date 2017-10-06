#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

FILE=act-detector-initial-caffemodels.tgz
CHECKSUM=d3347d864067ab3a42e3dc48fc50ed43

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Model checksum is correct. No need to download."
    exit 0
  else
    echo "Model checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading VGG16 caffemodels used as initialization for the RGB and the FLOW5 streams ..."

wget http://pascal.inrialpes.fr/data2/act-detector/downloads/initial_models/act-detector-initial-caffemodels.tgz

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."