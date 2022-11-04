#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`

WORKSPACE=${1:-"./workspaces/audioset_tagging"}   # Default argument.

CUDA_VISIBLE_DEVICES="1" nohup python3 pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='Cnn14' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='none' \
    --batch_size=512 \
    --num_workers=16 \
    --learning_rate=1e-3 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --n_epoches=2 \
    --sampler \
    --cuda \
    2>&1 | tee Logs/$timestamp.txt
