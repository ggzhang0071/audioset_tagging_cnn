# -*- coding: utf-8 -*-
# @Time    : 3/27/22 12:23 AM
# @Author  : Gege Zhang
import json,csv
from random import random
import torch
import librosa
from torch.utils.data import Dataset
import numpy as np
import random


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict ={}
    if not list_data_dict==[None]:    
        for key in list_data_dict[0].keys():
            tmp=np.array([data_dict[key] for data_dict in list_data_dict])
            if key!='audio_name':
                np_data_dict[key] = torch.from_numpy(tmp)
            else:
                np_data_dict[key] = tmp
        return np_data_dict



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
        param index: index of the audio file
        return: audio file, label, and index
        """
        # what is there means?
        datum = self.data[index]
        try:
            waveform, sample_rate=librosa.load(datum['wav'], sr=16000)
        except:
            print("{} error".format(datum['wav']))
        else:
            #padding the waveform to 16000
            if waveform.shape[0] < 16000:
                waveform1 = np.pad(waveform, (0, 16000 - waveform.shape[0]), 'constant')
            elif waveform.shape[0] >= 16000:
                waveform1 = waveform[:16000]
            # one-hot encoded label
            label_indices=np.zeros(self.label_num)
            label_indices[datum['labels']]=1
            float_label=label_indices.astype(np.float32)
            #label_indices=torch.FloatTensor(label_indices)
            #output_dict={"audio_name":datum['wav'],"waveform":waveform1, "target":float_label, "mixup_lambda":0}
            return  waveform1, float_label 
            
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    json_path='/git/datasets/from_audioset/datafiles_ok/part_audioset_train_data_1.json'
    label_num=4 
    """train_json_path='/git/datasets/from_audioset/datafiles_ok/part_audioset_train_data_1.json'
    train_dataset=AudiosetDataset(train_json_path,label_num)
    json_path='/git/datasets/from_audioset/datafiles_ok/part_audioset_eval_data_1.json'"""

    json_path='/git/datasets/from_audioset/datafiles_ok/chooosed_human_sounds.csv'



    val_dataset=AudiosetDataset(json_path,label_num)
    n_train = len(val_dataset)
    indices = random.sample(list(range(n_train)),100)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    #label_csv='/git/datasets/from_audioset/datafile_ok/part_audioset_class_labels_indices.csv'
    label_num=4
    dataset=AudiosetDataset(json_path,label_num)
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset,collate_fn=collate_fn, sampler=train_sampler)


    for i, data_dict in enumerate(eval_bal_loader):
        print(i)




            