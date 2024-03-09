import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMNetwork(nn.Module):

    def __init__(self, input_dims, n_classes, device, batch_first):
        #Model parameters
        hidden_dim = 128
        num_lstm_layers = 1

        #Set the device
        self.device = device

        #Set the batch_first
        self.batch_first = batch_first

        #Model nn module intialization
        super().__init__()
        self.n_layers = num_lstm_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dims, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, X_batch):
        output, (hidden, carry) = self.lstm(X_batch)
        #unpack output
        output, __ = pad_packed_sequence(output, batch_first=self.batch_first)
        return self.fc(output)