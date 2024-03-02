import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from data import EmotionDataset, collate_fn, EMOTION_IDX, IDX_2_EMOTION
import json
import re
from models import regression
from transformers import BertTokenizer, BertModel

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
    os.makedirs(save_path, exist_ok=True)
    torch.save(save_dict, save_path)


def process_data_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    proc_data = {}
    idx = 0

    for key in data.keys():
        keep_sample = True
        sample = data[key]
        sentence = sample["full sentence"]
        segments = sample["segments"]
        concat_segments = ""
        for seg in segments:
            if seg["Emotion"].upper() not in EMOTION_IDX:
                keep_sample = False
                break

            cur_segment = concat_segments + seg["Segment"]
            if cur_segment != sentence[:len(cur_segment)]:
                pattern = r'(?<=concat_segments).*?(?=seg["Segment"])'
                try:
                    match = re.findall(pattern, sentence)[0]
                except:
                    continue
                seg['segment'] = match + seg["Segment"]

            concat_segments += seg["Segment"]

        if concat_segments != sentence:
            keep_sample = False

        if keep_sample:
            proc_data[idx] = sample
            idx += 1

    with open(output_path, 'w') as f:
        json.dump(proc_data, f, indent=4)


def inference(config, dataset, device="cuda"):

    if config["data"]["use_start_end"]:
        n_classes = 3 * len(EMOTION_IDX)
    else:
        n_classes = len(EMOTION_IDX)

    data = {}

    # define model
    if config["model_name"] == "regression":
        model = regression.Regression(input_dims=config["data"]["bert_dim"], n_classes=n_classes)
    elif config["model_name"] == "lstm":
        raise NotImplementedError("Implementation not complete yet")
        model = lstm.LSTM(config)
    
    ckpt = torch.load(config["inference"]["checkpoint"])
    model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    model.eval()

    # define hugging face models for getting word vectors
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    for i in range(len(dataset)):
        sample = dataset[i]

        # get data from batch (dictionary)
        embs = sample["embeddings"].float()
        lengths = sample["num_tokens"].to(torch.long)
        labels = sample["labels"].to(torch.long)
        tokens = sample["tokens"].to(torch.long)

        # move to device
        embs = embs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # model predictions
        outputs = model(embs)

        segments = []
        emotions = []
        if config["data"]["use_start_end"]:
            pass
        else:
            pred_classes = torch.argmax(outputs, dim=1)
            seg = [tokens[0]]
            emt = IDX_2_EMOTION[pred_classes[0].cpu().item()]
            for j in range(1, lengths[0]):
                if pred_classes[j] == pred_classes[j - 1]:
                    seg.append(tokens[j])
                else:
                    segments.append(seg)
                    emotions.append(emt)
                    seg = [tokens[j]]
                    emt = IDX_2_EMOTION[pred_classes[j].cpu().item()]
        
        # decode tokens
        words = tokenizer.convert_ids_to_tokens(segments)

        data[str(i)] = {
            "num_segments": len(segments),
            "segments": [
                {"Segment": words[s], "Emotion": emotions[s]} for s in range(len(segments))
            ]
        }

        with open(config["inference"]["output_file"]) as f:
            json.dump(data, f)



            

