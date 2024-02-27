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
import yaml


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(config):

    train_loader = utils.create_dataloader(config, split="train")

    # define model
    if config["model_name"] == "regression":
        model = regression.Regression(config)
    elif config["model_name"] == "lstm":
        model = lstm.LSTM(config)

    model = model.to(device)

    # define loss function
    if config["loss_fn"] == "crossentropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function %s" % (config["training"]["loss_fn"]))
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    writer = SummaryWriter()

    if config["training"]["restore_ckpt"] is not None:
        ckpt = torch.load(config["restore_ckpt"])
        num_iters = ckpt['num_iters']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        run_name = ckpt['run_name']

        print("Restored checkpoint at iteration %d..." % (num_iters))
    else:
        print("Starting training at iteration 0...")
        num_iters = 0
        run_name = datetime.now().strftime('%d-%m-%Y-%H-%M')

    for e in range(config["num_epochs"]):
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
            if num_iters % config["log_freq"] == 0:
                writer.add_scalar("Loss/train", loss.item(), num_iters)

            
            if num_iters % config["val_freq"] == 0:
                # TODO: run validation code
                pass

            if num_iters % config["save_freq"] == 0:
                utils.save_state(config, model, optimizer, run_name, num_iters)

    # save at end of training
    if num_iters % config["save_freq"] == 0:
        utils.save_state(config, model, optimizer, run_name, num_iters)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config_file)
    if args.train:
        train(config)
    elif args.test:
        pass
        # call code to test provided model on test data

    

