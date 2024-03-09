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
from data import EMOTION_IDX, EmotionDataset
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train(config):

    _, train_loader = utils.create_dataloader(config, split="train", \
                pack_seq=config["data"]["pack_seq"], batch_first=config["data"]["batch_first"], \
                device=device)
    
    _, val_loader = utils.create_dataloader(config, split="val", \
                pack_seq=config["data"]["pack_seq"], batch_first=config["data"]["batch_first"], \
                device=device)
    
    # define number of classes
    if config["data"]["use_start_end"]:
        n_classes = 3 * len(EMOTION_IDX)
    else:
        n_classes = len(EMOTION_IDX)

    # define model
    if config["model_name"] == "regression":
        model = regression.Regression(input_dims=config["data"]["bert_dim"], n_classes=n_classes)
    elif config["model_name"] == "lstm":
        model = lstm.LSTMNetwork(input_dims=config["data"]["bert_dim"],\
            n_classes=n_classes, device=device, config=config)

    model = model.to(device)
    model.train()

    # define loss function
    if config["training"]["loss_fn"] == "crossentropy":
        # reduction is none because we need to average loss differently to account for padding
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError("Invalid loss function %s" % (config["training"]["loss_fn"]))
    
    # define optimizer
    if config["training"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    elif config["training"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["lr"], momentum=0.9)
    else:
        raise ValueError("Optimizer not defined %s" % (config["training"]["optimizer"]))

    if config["training"]["restore_ckpt"] is not None:
        ckpt = torch.load(config["training"]["restore_ckpt"], map_location=device)
        num_iters = ckpt['num_iters']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        run_name = ckpt['run_name']
        purge_step = num_iters

        print("Restored checkpoint at iteration %d..." % (num_iters))
    else:
        print("Starting training at iteration 0...")
        num_iters = 0
        if config["run_name"] is None:
            run_name = datetime.now().strftime('%d-%m-%Y-%H-%M')
        else:
            run_name = config["run_name"]
        purge_step = None

    # creating summary writer and runs and enabling restart of summary writer
    writer = SummaryWriter(log_dir=os.path.join("./runs", run_name), purge_step=purge_step)

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

            #Check if tehy are packed sequences, return just data
            if isinstance(embs, PackedSequence):
                embs, __ = pad_packed_sequence(embs, batch_first=config["data"]["batch_first"])
            if isinstance(lengths, PackedSequence):
                lengths, __ = pad_packed_sequence(lengths, batch_first=config["data"]["batch_first"])
            if isinstance(labels, PackedSequence):
                labels, ___ = pad_packed_sequence(labels, batch_first=config["data"]["batch_first"])

            loss_mask = (labels > 0).float()
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

            if num_iters % config["logging"]["val_freq"] == 0:
                print("running validation...")
                val_loss = utils.run_validation(val_loader, model, loss_fn, config=config, device=device)
                print("Validation loss at iter %d: %.3f" % (num_iters, val_loss))
                writer.add_scalar("Loss/val", val_loss, num_iters)

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
       dataset, _ = utils.create_dataloader(config, split="test", \
                   pack_seq=config["data"]["pack_seq"], batch_first=config["data"]["batch_first"], \
                   device=device)
       utils.inference(config, dataset, device=device)