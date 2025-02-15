import os,random
import sys
import numpy as np
import argparse
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim
import torch.utils.data
import h5py
#from  torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter()
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import (create_folder, get_filename, create_logging, Mixup, StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, AlternateTrainSampler, EvaluateSampler, collate_fn)
#from  data_generator import collate_fn
sys.path.append("/git/audioset_tagging_cnn")
#from utils import AudiosetDataset, collate_fn
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, 
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func

import torch.cuda.profiler as profiler
#import torch.cuda.nvtx as nvtx


def train(args):
    """Train AudioSet tagging model. 
    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """
    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    label_num=args.label_num
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    # Paths
    black_list_csv = None
    
    train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 'balanced_train.h5')

    eval_bal_indexes_hdf5_path = os.path.join(workspace,  'hdf5s', 'indexes', 'eval.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',  'test.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    #logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    """train_json_path='/git/datasets/from_audioset/datafiles_ok/part_audioset_train_data_1.json'
    train_dataset=AudiosetDataset(train_json_path,label_num)
    val_json_path='/git/datasets/from_audioset/datafiles_ok/part_audioset_eval_data_1.json'
    val_dataset=AudiosetDataset(val_json_path,label_num)
    test_json_path="/git/datasets/from_audioset/datafiles_ok/chooosed_human_sounds.csv"
    test_dataset=AudiosetDataset(test_json_path,label_num)"""


    """save_train_h5_path='/git/datasets/from_audioset/datafiles_ok/train_human_sound_dataset.h5'
    save_eval_h5_path='/git/datasets/from_audioset/datafiles_ok/eval_human_sound_dataset.h5'"""

    save_train_h5_path="/git/audioset_tagging_cnn/workspaces/audioset_tagging/hdf5s/waveforms/eval.h5"
    save_eval_h5_path="/git/audioset_tagging_cnn/workspaces/audioset_tagging/hdf5s/waveforms/test.h5"

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    
    # Evaluate sampler

    eval_bal_sampler = EvaluateSampler(indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)    

    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    """if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)"""

    # Evaluator
    evaluator = Evaluator(model=model)
        
    """# Statistics
    statistics_container = StatisticsContainer(statistics_path)"""
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        #statistics_container.load_state_dict(resume_iteration)
        #iteration = checkpoint['iteration']
    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
        
    
    time1 = time.time()
    """ if not isinstance(model,nn.DataParallel):
        model=nn.DataParallel(Model)   # get the model from the dataparallel
    model=model.to(device)"""

    pbar=tqdm(range(1,args.n_epoches+1))
    iter_to_capture=args.n_epoches//2
    with torch.autograd.profiler.emit_nvtx():
        for epoch in  pbar:
            if epoch==iter_to_capture:
                profiler.start()
                
            for batch_data_dict in eval_bal_loader:
                """ batch_data_dict: {
                    'audio_name': (batch_size [*2 if mixup],), 
                    'waveform': (batch_size [*2 if mixup], clip_samples), 
                    'target': (batch_size [*2 if mixup], classes_num), 
                    (ifexist) 'mixup_lambda': (batch_size * 2,)}
                """
                # Evaluate
                if (iteration % 10 == 0 and iteration > resume_iteration):
                    train_fin_time = time.time()
                    bal_statistics = evaluator.evaluate(eval_bal_loader)
                    test_statistics = evaluator.evaluate(eval_test_loader)
                                    
                    """logging.info("iteration: {}".format(iteration),'Validate bal mAP: {:.3f}'.format(np.mean(bal_statistics['average_precision'])), 
                        'Validate bal F1_score: {:.3f}'.format(np.mean(bal_statistics['F1_score'])), 
                        'Validate bal roc_auc: {:.3f}'.format(np.mean(bal_statistics['roc_auc'])),  """
                    print("\n epoch: {},".format(epoch),'Validate bal F1_score: {:.3f},'.format(np.mean(bal_statistics['F1_score'])), '\n Validate bal acc_score: {:.3f},'.format(np.mean(bal_statistics['acc_score'])),  'Validate bal recall_score: {:.3f}.'.format(np.mean(bal_statistics['recall_score'])))

                    """logging.info('Validate test mAP: {:.3f}'.format(
                        np.mean(test_statistics['average_precision'])),'Validate test F1_score: {:.3f}'.format(np.mean(test_statistics['F1_score']))
                        ,'Validate test roc_auc: {:.3f}'.format(np.mean(test_statistics['roc_auc'])),"""
                    print("\n epoch: {}".format(epoch),'Validate test F1_score: {:.3f},'.format(np.mean(test_statistics['F1_score'])),'Validate test acc_score: {:.3f},'.format(np.mean(test_statistics['acc_score']), 'Validate test recall_score: {:.3f}.'.format(np.mean(test_statistics['recall_score']))))    

                    train_bgn_time = time.time()

                    """statistics_container.append(iteration, bal_statistics, data_type='bal')
                    statistics_container.append(iteration, test_statistics, data_type='test')
                    statistics_container.dump()"""

                    train_time = train_fin_time - train_bgn_time
                    validate_time = time.time() - train_fin_time
                    """logging.info(
                        'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                            ''.format(iteration, train_time, validate_time))"""

                    logging.info('------------------------------------')

                    train_bgn_time = time.time()
            
                """# Save model
                if iteration % 100000 == 0:
                    checkpoint = {
                        'iteration': iteration, 
                        'model': model.module.state_dict(), 
                        #'sampler': train_sampler.state_dict()
                        }

                    checkpoint_path = os.path.join(
                        checkpoints_dir, '{}_iterations.pth'.format(iteration))
                        
                    torch.save(checkpoint, checkpoint_path)
                    #logging.info('Model saved to {}'.format(checkpoint_path))"""
                
                # Mixup lambda
                """if 'mixup' in augmentation:
                    batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                        batch_size=len(batch_data_dict['waveform']))"""

                # Move data to device
                """for key in batch_data_dict.keys():
                    batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)"""
                
                # Forward
                model.train()
                if 'mixup' in augmentation:
                    batch_output_dict = model(batch_data_dict[1].to(device,non_blocking=True), 
                        batch_data_dict['mixup_lambda'])
                    """{'clipwise_output': (batch_size, classes_num), ...}"""

                    batch_target_dict = {'target': do_mixup(batch_data_dict[2], 
                        batch_data_dict['mixup_lambda'])}
                    """{'target': (batch_size, classes_num)}"""
                else:
                    batch_output_dict = model(torch.from_numpy(batch_data_dict['waveform']).to(device,non_blocking=True), None)
                    #batch_output_dict = model(batch_data_dict[0].to(device,non_blocking=True), None)
                    """{'clipwise_output': (batch_size, classes_num), ...}"""

                    batch_target_dict = {'target': torch.from_numpy(batch_data_dict['target']).to(device,non_blocking=True)}
                    """{'target': (batch_size classes_num)}"""

                # Loss
                loss = loss_func(batch_output_dict, batch_target_dict)

                # Backward
                loss.backward()
                #print(loss.item())
                writer.add_scalar("train/loss", loss.item(), epoch)
                
                optimizer.step()
                optimizer.zero_grad()
                if epoch==iter_to_capture:
                    profiler.stop()
                
                if iteration % 10 == 0:
                    """print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                        .format(iteration, time.time() - time1))"""
                    time1 = time.time()
            
            # Stop learning
            if iteration == early_stop:
                break
            iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode') 
    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=16000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--n_epoches', type=int, default=100)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--num_workers',type=int,default=16)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=1000000)
    parser_train.add_argument('--cuda', action='store_true')
    parser_train.add_argument('--sampler', action='store_true')
    parser_train.add_argument('--label_num',type=int,default=4)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')