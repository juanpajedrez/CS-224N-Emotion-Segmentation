import numpy as np
import torch
import sys, os
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer, BertModel

# define class indices for task here
EMOTION_IDX = {"SAD": 0, "HAPPY": 1, "ANGER": 2, "SURPRISE": 3, "DISGUST": 4}

class EmotionDataset(Dataset):

    def __init__(self, params):
        super().__init__()
        self.params = params["dataset"]
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


# if __name__ == "__main__":    
#     data_path = os.path.join("./data", "dummy.json")

#     params_test = {"dataset": {"train_filepath": data_path, "use_start_end": True}}
#     emotion_dataset = EmotionDataset(params=params_test) 
#     breakpoint()