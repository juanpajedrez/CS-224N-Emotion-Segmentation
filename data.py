import numpy as np
import torch
import sys, os
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# define class indices for task here
EMOTION_IDX = {"ANGRY": 0, "SURPRISED": 1, "DISGUSTED": 2, "HAPPY": 3, "FEARFUL": 4, "SAD": 5, "NEUTRAL": 6}

class EmotionDataset(Dataset):

    def __init__(self, params):
        super().__init__()
        self.params = params["data"]
        with open(self.params['train_filepath']) as f:
            self.data = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_metadata = self.data[str(idx)]
        segments = file_metadata["segments"]
        segments_tokenized = []
        segments_embedded = []
        labels_list = []

        for  seg in segments:
            tokenized_seg = self.tokenizer(seg["Segment"], return_tensors = "pt")
            segments_tokenized.append(np.array(tokenized_seg["input_ids"]).reshape((-1,)))
            outputs = self.model(**tokenized_seg)
            embeddings = outputs.last_hidden_state[0]
            segments_embedded.append(embeddings)

            # labels
            labels = [EMOTION_IDX[seg["Emotion"].strip().upper()] for _ in range(len(tokenized_seg["input_ids"][0]))]
            if self.params["use_start_end"]:
                labels = [3 * x + 1 for x in labels]
                labels[0] = labels[0] - 1
                labels[-1] = labels[-1] + 1
            labels_list.append(np.array(labels).reshape((-1,)))
        
        tokens = torch.tensor(np.concatenate(segments_tokenized)).squeeze(0)
        labels = torch.tensor(np.concatenate(labels_list))
        embeddings = torch.cat(segments_embedded)
        num_tokens = torch.tensor(len(tokens))

        return {"tokens": tokens, "labels": labels, "embeddings": embeddings, "num_tokens": num_tokens}


# collate function to collate samples into a batch and pad (pack if specified) as needed
class collate_fn:

    def __init__(self, pack_seq=False, batch_first=True):
        self.pack_seq = pack_seq
        self.batch_first = batch_first

    def __call__(self, batch):
        length_list = []
        emb_list = []
        label_list = []
        for sample in batch:
            length_list.append(sample['num_tokens'])
            emb_list.append(sample['embeddings'])
            label_list.append(sample['labels'])

        lengths = torch.stack(length_list)
        padded_embs = pad_sequence(emb_list, batch_first=self.batch_first, padding_value=0)
        padded_labels = pad_sequence(label_list, batch_first=self.batch_first, padding_value=-1)
        if self.pack_seq:
            padded_embs = pack_padded_sequence(padded_embs, lengths, batch_first=self.batch_first)
            padded_labels = pack_padded_sequence(padded_labels, lengths, batch_first=self.batch_first)

        proc_batch = {
            "embeddings": padded_embs,
            "labels": padded_labels,
            "num_tokens": lengths
        }

        return proc_batch

#if __name__ == "__main__":    
#    data_path = os.path.join("./data", "dummy.json")
#
#    params_test = {"data": {"train_filepath": data_path, "use_start_end": True}}
#    emotion_dataset = EmotionDataset(params=params_test) 
#    breakpoint()