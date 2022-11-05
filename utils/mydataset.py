import argparse
import os, json, h5py
import config
import numpy  as np
import librosa
from utilities import pad_or_truncate, float32_to_int16, create_logging
import logging

def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """
    waveforms_hdf5_path = args.waveforms_hdf5_path
    json_path = args.json_path
    sample_rate=config.sample_rate

    clip_samples=config.clip_samples

     #create dir
    waveforms_dir=os.path.dirname(waveforms_hdf5_path)
    if not os.path.exists(waveforms_dir):
        os.makedirs(waveforms_dir)
    elif os.path.exists(waveforms_hdf5_path):
        os.remove(waveforms_hdf5_path)

    with open(json_path,"r") as f:
        train_data = json.load(f)
        meta_data=train_data['data']
    # crete hdf5 file
    audios_num=len(meta_data)
    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        # create dataset
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S100')
        hf.create_dataset('waveform',shape=(audios_num, clip_samples), dtype=np.int16)
        hf.create_dataset('target', shape=(audios_num,config.classes_num),  dtype=bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)
        # pack waveform & target of several audio clips to a single hdf5 file
        # write data
        for n,  wav_label in enumerate(meta_data):
            audio_path=wav_label["wav"]
            if os.path.isfile(audio_path):
                audio,_=librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio=pad_or_truncate(audio, clip_samples)
                hf["audio_name"][n]=audio_path.encode()
                hf["waveform"][n]=float32_to_int16(audio)
                hf["target"][n]=wav_label["labels"]
            else:
                print("file not exist:{}".format(audio_path))
    print("write hdf5 file to {}".format(waveforms_hdf5_path))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    subparsers=parser.add_subparsers(dest='mode')
    parser_pack_wavs=subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack_wavs.add_argument('--json_path', type=str, default='data/train.json')
    parser_pack_wavs.add_argument('--audios_dir', type=str)
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, default='data/waveforms.h5')
    args=parser.parse_args()
    if args.mode=='pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)
    else:
        raise Exception("Error")


        



    


    

        



