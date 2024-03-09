import torch
from torch import nn
import numpy as np

class LSTMNetwork(nn.Module):

    def __init__(self, input_dims, n_classes, device):
        #Model parameters
        hidden_dim = 128
        num_lstm_layers = 1

        #Set the device
        self.device = device

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
        hidden, carry = (
            torch.randn(self.n_layers, len(X_batch), self.hidden_dim).to(self.device), 
            torch.randn(self.n_layers, len(X_batch), self.hidden_dim).to(self.device),
        )
        output, (hidden, carry) = self.lstm(X_batch, (hidden, carry))
        return self.fc(output[:,-1])