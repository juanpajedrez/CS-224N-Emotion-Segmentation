import torch
import numpy as np
import torch.nn as nn

class NGram(nn.Module):

    def __init__(self, params, emotion_embs, device):
        super().__init__()

        self.params = params
        self.num_segments = params["inference"]["ngram"]
        assert self.num_segments is not None, "num_segments in ngram model is None"
        self.device = device

        self.emotion_embs = emotion_embs
        # normalizing emotion embs
        self.emotion_embs = emotion_embs / torch.norm(emotion_embs, dim=-1, keepdim=True).to(device)


    @torch.no_grad()
    def forward(self, x, num_tokens):

        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            num_tokens = num_tokens.view(1, -1)

        idx_start = torch.zeros_like(num_tokens)
        step = num_tokens // self.num_segments
        idx_end = step

        # normalizing for emotion
        x = x / torch.norm(x, dim=-1, keepdim=True)

        preds = torch.zeros(x.shape[0], self.num_segments, device=self.device)
    
        for i in range(self.num_segments):
            
            # defining mask to find average embedding
            mask = torch.zeros(x.shape[0], x.shape[1] + 1, device=self.device, dtype=x.dtype)
            batch_arr = torch.arange(x.shape[0], device=x.device)
            mask[batch_arr, idx_start] = 1
            mask[batch_arr, idx_end] = 1
            mask = torch.cumsum(mask, dim=1)
            mask = (mask != 1)
            mask = mask[..., :x.shape[1]]
            mask = mask.unsqueeze(2)
            mask = mask.expand(-1, -1, x.shape[2])

            # avg embedding of segment (assuming batch first)
            seg = x.clone()
            seg[mask] = 0
            denom = (idx_end - idx_start)
            seg = torch.sum(seg, dim=1) / denom

            # find closest emotions using cosine similarity
            seg = seg / torch.norm(seg, dim=-1, keepdim=True)
            cos_sim = torch.mm(seg, self.emotion_embs.T)
            pred_emt = torch.argmax(cos_sim, dim=-1)
            preds[:, i] = pred_emt
            
            # update indices for next segments
            idx_start = idx_end
            if i == self.num_segments - 2:
                idx_end = num_tokens
            else:
                idx_end = idx_end + step

        return preds


