import argparse
import torch
import random
from models import Cnn14
import torch.nn as nn
import torch.optim as optim
from evaluate import Evaluator
import sys, os 
import h5py
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, AlternateTrainSampler, EvaluateSampler, collate_fn)
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from evaluate import Evaluator
import logging


def train(args):
    device=torch.device("cuda" if args.cuda and torch.cuda.is_available() else torch.device("cpu")) 
    batch_size=args.batch_size
    num_workers=args.num_workers
    classes_num=args.classes_num 

    # add model
    Model=eval(args.model_type)
    model= Model(sample_rate=args.sample_rate, window_size=args.window_size, 
        hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax, 
        classes_num=classes_num)
    if not isinstance(model,nn.DataParallel):
        model=nn.DataParallel(model)   # get the model from the dataparallel
    model.cuda()
    if "cuda" in str(device):
        logging.info("using GPU num:{}".format(torch.cuda.device_count()))
    else:
        logging.info("using CPU")

    # add optimizer
     # Evaluator
    evaluator = Evaluator(model=model)
    #optimizer= optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-08, weight_decay=0., amsgrad=True)


    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    # paths
    black_list_csv = None
    indexes_hdf5_dir="/git/audioset_tagging_cnn/workspaces/audioset_tagging/hdf5s/indexes"

    train_bal_indexes_hdf5_path=os.path.join(indexes_hdf5_dir,'balanced_train.h5')
    eval_bal_indexes_hdf5_path=os.path.join(indexes_hdf5_dir,'eval.h5')
    eval_test_indexes_hdf5_path=os.path.join(indexes_hdf5_dir,'test.h5')

    train_sampler=BalancedTrainSampler(indexes_hdf5_path=train_bal_indexes_hdf5_path, 
        batch_size=batch_size, black_list_csv=black_list_csv)
    eval_sampler=EvaluateSampler(indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)
    test_sampler=EvaluateSampler(indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    dataset=AudioSetDataset(sample_rate=args.sample_rate)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=train_sampler, collate_fn=collate_fn,  num_workers=num_workers, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=eval_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=test_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    
    pbar=tqdm(range(args.n_epoches))
    scalar=GradScaler()
    for epoch in pbar:
        for batch_data_dict in train_loader:
            # get the data  
            waveforms=batch_data_dict['waveform'].to(device,non_blocking=True)
            targets=batch_data_dict['target'].to(device,non_blocking=True)
            # forward
            optimizer.zero_grad()
            with autocast(): 
                batch_output_dict=model(waveforms)
                loss=F.binary_cross_entropy(batch_output_dict['clipwise_output'],targets)
            # backward
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            print(loss.item())
        if epoch%10==0:
            validate(model, eval_bal_loader,device)
        

def validate(model,eval_bal_loader,device):
    all_predictions=[]
    all_targets=[]
    with torch.no_grad():
        for batch_data_dict in eval_bal_loader:
            """ batch_data_dict: {
                    'audio_name': (batch_size [*2 if mixup],), 
                    'waveform': (batch_size [*2 if mixup], clip_samples), 
                    'target': (batch_size [*2 if mixup], classes_num), 
                    (ifexist) 'mixup_lambda': (batch_size * 2,)}
            """
            waveforms=batch_data_dict['waveform'].to(device,non_blocking=True)
            targets=batch_data_dict['target'].to(device,non_blocking=True)
            batch_output_dict=model(waveforms)
            clipwise_output=batch_output_dict['clipwise_output'].cpu().numpy()
            # compute the loss

            val_loss = F.binary_cross_entropy(clipwise_output,targets)
            bal_statistics=eva
            predicted_label=np.argmax(clipwise_output, axis=1)

            target=batch_data_dict['target'].numpy()

            target_label=np.argmax(target, axis=1)
            all_predictions.append(predictions)
            all_targets.append(target_label)



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode') 
    parser_train = subparsers.add_parser('train') 
    #parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, default="Cnn14")
    parser_train.add_argument('--sample_rate', type=int, default=16000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000)
    parser_train.add_argument('--classes_num', type=int, default=4)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--n_epoches', type=int, default=4)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--num_workers', type=int, default=4)
    parser_train.add_argument('--sampler', action='store_false')
    parser_train.add_argument('--cuda', action='store_false')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception("Error!")


    