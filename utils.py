import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from data import EmotionDataset, collate_fn, EMOTION_IDX
import json
import re

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
                pattern = r'(?<=START).*?(?=END)'
                match = re.findall(pattern, text)
                seg['segment'] = sentence[len(concat_segments):]

            concat_segments += seg["Segment"]


        if concat_segments != sentence:
            keep_sample = False

        if keep_sample:
            proc_data[idx] = sample
            idx += 1

    with open(output_path, 'w') as f:
        json.dump(proc_data, f, indent=4)


# if __name__=="__main__":
#     process_data_json("./data/0229_formatted.json", "./data/0229_formatted_proc.json")