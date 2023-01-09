# create HDF5 dataset of wav files
import h5py
import json, os 
import librosa
import numpy as np


def get_wave_list(json_file):
    with open(json_file,"r") as f:
        train_data = json.load(f)
        train_data_list=train_data['data']
        waves, labels=[],[]
        counter=0
        for wav_label in train_data_list:
            if counter<10:
                try:
                    waveform, sample_rate=librosa.load(wav_label["wav"], sr=16000)
                except:
                    print("{} error".format(wav_label["wav"]))
                else:  
                    #padding the waveform to 16000
                    """if waveform.shape[0] < 16000:
                        waveform1 = np.pad(waveform, (0, 16000 - waveform.shape[0]), 'constant')
                    elif waveform.shape[0] >= 16000:
                        waveform1 = waveform[:16000]"""
                    waves.append(wav_label["wav"])
                    labels.append(wav_label["labels"])
                    counter+=1
            else:
                return waves, labels

if __name__ == "__main__":

    wav_dir="/git/datasets/from_audioset"
    json_dir="/git/datasets/from_audioset/datafiles_ok"
    train_json_file="part_audioset_train_data_1.json"
    eval_json_file="part_audioset_eval_data_1.json" 
    save_train_h5_path='/git/datasets/from_audioset/datafiles_ok/train_human_sound_dataset.h5'
    save_eval_h5_path='/git/datasets/from_audioset/datafiles_ok/eval_human_sound_dataset.h5'
    batch_size=1000

    train_wav_list, train_label_list=get_wave_list(os.path.join(json_dir,train_json_file))
    eval_wav_list, eval_label_list=get_wave_list(os.path.join(json_dir,eval_json_file))


    with h5py.File(save_train_h5_path, 'w') as f:
        train_wav=f.create_dataset("audio_name", data=train_wav_list, dtype='f',compression="gzip", compression_opts=9,chunks=True)
        train_label_list=f.create_dataset("target", data=train_label_list, dtype='f',compression="gzip", compression_opts=9,chunks=True)
    
    with h5py.File(save_eval_h5_path, 'w') as f:
        eval_wav=f.create_dataset("audio_name", data=eval_wav_list, dtype='f',compression="gzip", compression_opts=9,chunks=True)
        eval_label_list=f.create_dataset("target", data=eval_label_list, dtype='f',compression="gzip", compression_opts=9,chunks=True)
    
    with h5py.File(save_train_h5_path, 'r') as f:
        print(f["audio_name"].shape)
        print(f["target"].shape)
        
    with h5py.File(save_eval_h5_path,"r") as f:
        print(f["audio_name"].shape)
        print(f["target"].shape)

