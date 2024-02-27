import torch
import numpy as np
import os
import sys
import argparse
import utils
from tqdm import tqdm
from models import lstm, regression
from torch.utils.tensorboard import SummaryWriter
import datetime


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(args):

    train_loader = utils.create_dataloader(args, split="train")

    # define model
    if args.model_name == "regression":
        model = regression.Regression(args)
    elif args.model_name == "lstm":
        model = lstm.LSTM(args)

    model = model.to(device)

    # define loss function
    if args.loss_fn == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function %s" % (args.loss_fn))
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter()

    if args.restore_ckpt is not None:
        ckpt = torch.load(args.restore_ckpt)
        num_iters = ckpt['num_iters']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        run_name = ckpt['run_name']

        print("Restored checkpoint at iteration %d..." % (num_iters))
    else:
        print("Starting training at iteration 0...")
        num_iters = 0
        run_name = datetime.now().strftime('%d-%m-%Y-%H-%M')

    for e in range(args.num_epochs):
        for i, batch in enumerate(train_loader):

            # get data from batch (dictionary) and move to device
            x = None
            target = None

            # model predictions
            preds = model(x)

            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iters += 1
            if num_iters % args.log_freq == 0:
                writer.add_scalar("Loss/train", loss.item(), num_iters)

            
            if num_iters % args.val_freq == 0:
                # TODO: run validation code
                pass


            if num_iters % args.save_freq == 0:
                utils.save_state(args, model, optimizer, run_name, num_iters)

    # save at end of training
    if num_iters % args.save_freq == 0:
        utils.save_state(args, model, optimizer, run_name, num_iters)
                
        



    

