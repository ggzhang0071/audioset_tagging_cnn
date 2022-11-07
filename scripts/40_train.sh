#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`

WORKSPACE=${1:-"./workspaces/audioset_tagging"}   # Default argument.

CUDA_VISIBLE_DEVICES="1"  python3 -m pdb  pytorch/main1.py train \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='Cnn14' \
    --batch_size=512 \
    --num_workers=16 \
    --learning_rate=1e-3 \
    --n_epoches=2 \
    --sampler \
    --cuda \
    2>&1 | tee Logs/$timestamp.txt
