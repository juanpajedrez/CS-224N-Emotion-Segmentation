import torch
from torch.utils.data import DataLoader
import numpy as np
import os


def create_dataloader(args, split):
    raise NotImplementedError("Need to load dataset object")
    dataset = EmotionDataset(args, split=split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def save_state(args, model, optimizer, run_name, num_iters):
    save_dict = {}
    save_dict['num_iters'] = num_iters
    save_dict['optim_state_dict'] = optimizer.state_dict()
    save_dict['model_state_dict'] = model.state_dict()
    save_dict['run_name'] = run_name
    save_path = os.path.join(args.ckpt_dir, args.model_name, run_name, 'iter%d.pt' % (num_iters))
    torch.save(save_dict, os.path.join(args.ckpt_dir, args.model_name, ))