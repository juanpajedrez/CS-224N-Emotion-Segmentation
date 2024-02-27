import torch
from torch import nn
import numpy as np

class Regression(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Linear(args.input_dims, args.n_classes)

    def forward(self, x):
        preds = self.model(self.flatten(x))
        return preds