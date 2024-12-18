import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import torch.nn.functional as F


class LSTMNetwork(nn.Module):

    def __init__(self, input_dims, n_classes, device, config):
        #Model parameters
        hidden_dim = config["model"]["hidden_dim"]
        num_lstm_layers = config["model"]["num_layers"]
        bidir_flag = config["model"]["bidirectional"]
        bidir_multiplier = 2 if bidir_flag else 1

        #Set the device
        self.device = device

        #Set the batch_first
        self.batch_first = config["data"]["batch_first"]

        #Model nn module intialization
        super().__init__()
        self.n_layers = num_lstm_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dims, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=bidir_flag
        )

        self.fc = nn.Linear(bidir_multiplier * hidden_dim, n_classes)
    
    def forward(self, X_batch):
        output, (hidden, carry) = self.lstm(X_batch)
        
        #unpack output
        if isinstance(output, PackedSequence):
            output, __ = pad_packed_sequence(output, batch_first=self.batch_first)
        return self.fc(F.relu(output))