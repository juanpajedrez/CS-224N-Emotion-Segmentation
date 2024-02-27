import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# define class indices for problem here
EMOTION_IDX = {"SAD": 0, "HAPPY": 1, "ANGER": 2, "SURPRISE": 3, "DISGUST": 4}
CLASS_IDX = {
    "SAD-START": 0,
    "SAD-MIDDLE": 1,
    "SAD-END": 2,
    "HAPPY-START": 3,
    "HAPPY-MIDDLE": 4,
    "HAPPY-END": 5,
    "ANGER-START": 6,
    "ANGER-MIDDLE": 7,
    "ANGER-END": 8,
    "SURPRISE-START": 9,
    "SURPRISE-MIDDLE": 10,
    "SURPRISE-END": 11,
    "DISGUST-START": 12,
    "DISGUST-MIDDLE": 13,
    "DISGUST-END": 14
}


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