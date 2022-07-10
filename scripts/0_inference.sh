#!/bin/bash

# ------ Inference audio tagging result with pretrained model. ------
MODEL_TYPE="Cnn14"
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"

test_wav="resources/335.wav"

if [ -e CHECKPOINT_PATH ]; then
    echo "The file isn't existed, download"
    # Download audio tagging checkpoint.
   wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
else
    echo "The file is existed"
fi  
# Inference.
python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path=$test_wav \
    --cuda

# ------ Inference sound event detection result with pretrained model. ------
MODEL_TYPE="Cnn14_DecisionLevelMax"
CHECKPOINT_PATH="Cnn14_DecisionLevelMax_mAP=0.385.pth"
if [ -e CHECKPOINT_PATH ]; then
    echo "The file isn't existed, download"
    # Download sound event detection checkpoint.
    wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"

else
    echo "The file is existed"
fi

# Download sound event detection checkpoint.

# Inference.
python3 pytorch/inference.py sound_event_detection \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path=$test_wav\
    --cuda
    