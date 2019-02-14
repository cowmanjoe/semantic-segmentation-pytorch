#!/bin/bash

# Image and model names
H5_PATH=houses_256x256.hdf5
H5_URL=https://www.dropbox.com/s/lstcc2407whgrqw/houses_256x256?dl=0
MODEL_PATH=baseline-resnet50dilated-ppm_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $H5_PATH ]; then
  wget -O $H5_PATH $H5_URL
fi


# Inference
python3 -u batch_segmenter.py \
  --model_path $MODEL_PATH \
  --h5_path $H5_PATH \
  --arch_encoder resnet50dilated \
  --arch_decoder ppm_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH
