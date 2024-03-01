import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from data import EmotionDataset, collate_fn


def create_dataloader(args, split, pack_seq=False, batch_first=True):
    # pack_seq: to pack sequence in collate function, batch_first by default

    # TODO: support splitting of dataset into train/val/test
    dataset = EmotionDataset(args)
    dataloader = DataLoader(dataset, batch_size=args["training"]["batch_size"], shuffle=True, \
                    collate_fn=collate_fn(pack_seq=pack_seq, batch_first=batch_first))
    return dataloader


def save_state(args, model, optimizer, run_name, num_iters):
    save_dict = {}
    save_dict['num_iters'] = num_iters
    save_dict['optim_state_dict'] = optimizer.state_dict()
    save_dict['model_state_dict'] = model.state_dict()
    save_dict['run_name'] = run_name
    save_path = os.path.join(args["logging"]["ckpt_dir"], args["model_name"], run_name, 'iter%d.pt' % (num_iters))
    torch.save(save_dict, save_path)