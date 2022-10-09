# -*- coding: utf-8 -*-
# @Time    : 3/27/22 12:23 AM
# @Author  : Gege Zhang
import json,csv
from random import random
import torch
import torchaudio   
import librosa
from torch.utils.data import Dataset
import numpy as np
import config

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file,label_num):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        self.label_num=label_num
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        
    
    def __getitem__(self, index):
        """
        :param index: index of the audio file
        :return: audio file, label, and index
        """
        # what is there means?
        datum = self.data[index]
        try:
            waveform, sample_rate=librosa.load(datum['wav'], sr=16000)
        except:
            print(datum['wav'])
            print("error")
        else:
            waveform =waveform - waveform.mean()
            #padding the waveform to 16000
            if waveform.shape[0] < 16000:
                waveform1 = np.pad(waveform, (0, 16000 - waveform.shape[0]), 'constant')
            elif waveform.shape[0] > 16000:
                waveform1 = waveform[:16000]
            # one-hot encoded label
            label_indices=np.zeros(self.label_num)
            label_indices[datum['labels']]=1
            label_indices=torch.FloatTensor(label_indices)
            if waveform1.shape[0] != 16000:
                print(waveform1.shape,"error")
            output_dict={"audio_name":datum['wav'],"waveform":waveform1, "target":datum['labels']}
            return waveform1
            
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    json_path='/git/datasets/from_audioset/datafiles/part_audioset_eval_data_1.json'
    #label_csv='/git/datasets/from_audioset/datafiles/part_audioset_class_labels_indices.csv'
    label_num=4
    dataset=AudiosetDataset(json_path,label_num)
    for data_dict in dataset:
        pass 





            