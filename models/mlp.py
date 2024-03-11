import torch
from torch import nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_dims, n_classes):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dims, input_dims // 2),
                                   nn.ReLU(),
                                   nn.Linear(input_dims // 2, input_dims // 4),
                                   nn.ReLU(),
                                   nn.Linear(input_dims // 4, input_dims // 4),
                                   nn.ReLU(),
                                   nn.Linear(input_dims // 4, n_classes))

    def forward(self, x):
        preds = self.model(x)
        return preds