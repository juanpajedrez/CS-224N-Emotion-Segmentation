import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from data import EmotionDataset, collate_fn, EMOTION_IDX, IDX_2_EMOTION
import json
import re
from models import regression, mlp, ngram
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

BERT_IGNORE_TOKENS = [101, 102] # 101 is [CLS] and 102 is [SEP] for BERT
BERT_APOSTROPHE_TOKEN = 112

def create_dataloader(args, split, pack_seq=False, batch_first=True, device="cuda"):
    # pack_seq: to pack sequence in collate function, batch_first by default

    dataset = EmotionDataset(args, device=device)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.75, 0.05, 0.20], \
                                                                             generator=generator)
    if split == "train":
        dataset = train_dataset
    elif split == "val":
        dataset = val_dataset
    elif split == "test":
        dataset = test_dataset
    else:
        raise ValueError("Invalid split %s for creating dataset" % (split))

    dataloader = DataLoader(dataset, batch_size=args["training"]["batch_size"], shuffle=True, \
                    collate_fn=collate_fn(pack_seq=pack_seq, batch_first=batch_first))
    return dataset, dataloader


def save_state(args, model, optimizer, run_name, num_iters):
    save_dict = {}
    save_dict['num_iters'] = num_iters
    save_dict['optim_state_dict'] = optimizer.state_dict()
    save_dict['model_state_dict'] = model.state_dict()
    save_dict['run_name'] = run_name
    save_path = os.path.join(args["logging"]["ckpt_dir"], args["model_name"], run_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(save_dict, os.path.join(save_path, 'iter%d.pt' % (num_iters)))


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
    print("running inference!")

    if config["data"]["use_start_end"]:
        n_classes = 3 * len(EMOTION_IDX)
    else:
        n_classes = len(EMOTION_IDX)

    data = {}

    # define model
    if config["model_name"] == "regression":
        model = regression.Regression(input_dims=config["data"]["bert_dim"], n_classes=n_classes)
    elif config["model_name"] == "mlp":
        model = mlp.MLP(input_dims=config["data"]["bert_dim"], n_classes=n_classes)
    elif config["model_name"] == "ngram":
        emt_embs = dataset.dataset.compute_emotion_embs()
        model = ngram.NGram(config, emt_embs, device=device)
    elif config["model_name"] == "lstm":
        raise NotImplementedError("Implementation not complete yet")
        model = lstm.LSTM(config)
    
    if config["inference"]["checkpoint"] is not None:
        ckpt = torch.load(config["inference"]["checkpoint"])
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Warning: no checkpoint loaded")

    model = model.to(device)
    model.eval()
    
    tokenizer = dataset.dataset.tokenizer # dataset is a subset object

    gt_data = {}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        gt_data[str(i)] = dataset.dataset.data[str(dataset.indices[i])]

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
        if config["model_name"] == "ngram":
            outputs = model(embs, lengths)[0]
        else:
            outputs = model(embs)

        segments = []
        emotions = []
        if config["data"]["use_start_end"]:
            raise NotImplementedError("Need to implement inference when predicting middle token")
        elif config["model_name"] == "ngram":
            idx_start = 0
            num_segs = config["inference"]["ngram"]
            step = lengths.cpu().item() // num_segs
            idx_end = step
            for j in range(num_segs):
                segments.append([tokens[idx].cpu().item() for idx in range(idx_start, idx_end)])
                emt = IDX_2_EMOTION[int(outputs[j].cpu().item())]
                emotions.append(emt)
                idx_start = idx_end
                if j == num_segs - 2:
                    idx_end = lengths.cpu().item()
                else:
                    idx_end = idx_end + step
        else:
            pred_classes = torch.argmax(outputs, dim=1)
            seg = [tokens[0].cpu().item()]
            emt = IDX_2_EMOTION[pred_classes[0].cpu().item()]
            for j in range(1, lengths):
                if pred_classes[j] == pred_classes[j - 1]:
                    seg.append(tokens[j].cpu().item())
                else:
                    segments.append(seg)
                    emotions.append(emt)
                    seg = [tokens[j].cpu().item()]
                    emt = IDX_2_EMOTION[pred_classes[j].cpu().item()]
            segments.append(seg)
            emotions.append(emt)


        # decode tokens
        # WARNING: decoding doesn't preserve original sentence
        words = []
        ft_emotions = []
        for seg, emt in zip(segments, emotions):
            filtered_seg = [x for x in seg if x not in BERT_IGNORE_TOKENS]
            if len(filtered_seg) == 0:
                continue
            # decoded_seg = tokenizer.convert_ids_to_tokens(filtered_seg)
            # words.append(" ".join(decoded_seg))
            decoded_seg = tokenizer.decode(filtered_seg)
            # replace ## in decoded seg
            decoded_seg = decoded_seg.replace("##", "")
            words.append(decoded_seg)
            ft_emotions.append(emt)

        data[str(i)] = {
            "num_segments": len(words),
            "segments": [
                {"Segment": words[s], "Emotion": ft_emotions[s]} for s in range(len(words))
            ]
        }

    with open(config["inference"]["output_file"], 'w') as f:
        json.dump(data, f, indent=4)

    if config["inference"]["gt_json"] is not None:
        with open(config["inference"]["gt_json"], 'w') as f:
            json.dump(gt_data, f, indent=4)

@torch.no_grad()          
def run_validation(val_loader, model, loss_fn, device='cuda'):
    model.eval()
    running_loss = 0.0
    num_items = 0.0
    for i, batch in tqdm(enumerate(val_loader)):
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

        loss_mask = (labels > 0).float()
        labels[labels < 0] = 0
        loss = loss_fn(preds.permute(0, 2, 1), labels)
        loss = torch.sum(loss_mask * loss)

        running_loss += loss.cpu().item()
        num_items += torch.sum(loss_mask).cpu().item()

    avg_loss = running_loss / num_items
    model.train()
    return avg_loss