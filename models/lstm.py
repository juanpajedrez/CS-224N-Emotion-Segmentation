import torch
from torch import nn
import numpy as np

class LSTMNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.lstm = torch.nn.LSTM(self, args.input_dims, args.hidden_size, num_layers=args.n_lstm, \
                                  batch_first=False, dropout=args.dropout_p, \
                                  bidirectional=args.bidirectional, proj_size=0)
        

    def forward(self, x):
        preds = self.model(self.flatten(x))
        return preds