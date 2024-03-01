import torch
from torch import nn
import numpy as np

class Regression(nn.Module):

    def __init__(self, input_dims, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Linear(input_dims, n_classes)

    def forward(self, x):
        preds = self.model(x)
        return preds