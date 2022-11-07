#!/bin/bash
DATASET_DIR=${1:-"./datasets/audioset201906"}   # Default first argument.
WORKSPACE=${2:-"./workspaces/audioset_tagging"}   # Default second argument.


json_dir="/git/datasets/from_audioset/datafiles_ok"
train_json_file="part_audioset_train_data_1.json"
eval_json_file="part_audioset_eval_data_1.json" 
test_json_file="chooosed_test_human_sounds.json"

# Pack eval training waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$json_dir"/"$train_json_file \
    --audios_dir="/git/datasets/from_audioset" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" 2>&1 | tee $WORKSPACE"/hdf5s/waveforms/balanced_train.log"

<<COMMENT

# Pack test training waveforms to a single hdf5 file
python3 utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$json_dir"/"$test_json_file \
    --audios_dir="/git/datasets/from_audioset" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/test.h5" 2>&1 | tee $WORKSPACE"/hdf5s/waveforms/balanced_train.log"

# Pack evaluation waveforms to a single hdf5 file
python3   utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$json_dir"/"$eval_json_file \
    --audios_dir="/git/datasets/from_audioset" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"
COMMENT
<<COMMENT

# Pack unbalanced training waveforms to hdf5 files. Users may consider 
# executing the following commands in parallel to speed up. One simple 
# way is to open 41 terminals and execute one command in one terminal.


for IDX in {00..40}; do
    echo $IDX
    python3 utils/dataset.py pack_waveforms_to_hdf5 \
        --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" \
        --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" \
        --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" 2>&1 | tee $WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.log"
done
COMMENT