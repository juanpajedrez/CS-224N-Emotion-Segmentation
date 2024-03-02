import torch
import numpy as np
import os
import sys
import argparse
import utils
from tqdm import tqdm
from models import lstm, regression
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml
from data import EMOTION_IDX


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(config):

    train_loader = utils.create_dataloader(config, split="train", \
                pack_seq=config["data"]["pack_seq"], batch_first=config["data"]["batch_first"])
    
    # define number of classes
    if config["data"]["use_start_end"]:
        n_classes = 3 * len(EMOTION_IDX)
    else:
        n_classes = len(EMOTION_IDX)

    # define model
    if config["model_name"] == "regression":
        model = regression.Regression(input_dims=config["data"]["bert_dim"], n_classes=n_classes)
    elif config["model_name"] == "lstm":
        raise NotImplementedError("Implementation not complete yet")
        model = lstm.LSTM(config)

    model = model.to(device)

    # define loss function
    if config["training"]["loss_fn"] == "crossentropy":
        # reduction is none because we need to average loss differently to account for padding
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
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

    for e in range(config["training"]["num_epochs"]):
        for i, batch in tqdm(enumerate(train_loader)):

            # get data from batch (dictionary)
            embs = batch["embeddings"].float()
            lengths = batch["num_tokens"].to(torch.long)
            labels = batch["labels"].to(torch.long)

            # move to device
            embs = embs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            # model predictions
            preds = model(embs)

            loss_mask = (labels < 0).float()
            labels[labels < 0] = 0
            loss = loss_fn(preds.permute(0, 2, 1), labels)
            loss = torch.sum(loss_mask * loss) / torch.sum(loss_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iters += 1
            if num_iters % config["logging"]["log_freq"] == 0:
                writer.add_scalar("Loss/train", loss.item(), num_iters)
                print("Iteration %d, loss %.3f" % (num_iters, loss.item()))

            
            # if num_iters % config["logging"]["val_freq"] == 0:
            #     # TODO: run validation code
            #     pass

            if num_iters % config["logging"]["save_freq"] == 0:
                utils.save_state(config, model, optimizer, run_name, num_iters)

    # save at end of training
    if num_iters % config["logging"]["save_freq"] == 0:
        utils.save_state(config, model, optimizer, run_name, num_iters)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    if args.train:
        train(config)
    elif args.test:
        pass
        # call code to test provided model on test data

    

