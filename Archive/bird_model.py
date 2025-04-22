import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import rasterio
import fiona
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torcheval.metrics as metrics

import matplotlib.pyplot as plt

def infer_flattened_size(model, input_shape):
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        out = model(dummy)
        return out.view(1, -1).shape[1]

class birdNN(nn.Module):
    def __init__(self, num_layers, dropout_rate, kernelsize, pad, stride_len, 
                 input_dim, hidden_dim, output_dim):
        super(birdNN, self).__init__()
        ## Convolution Layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(input_dim, 16,
                                        kernel_size=kernelsize, stride=stride_len,
                                        padding=pad))
        self.conv_layers.append(nn.Conv2d(16, 32,
                                        kernel_size=kernelsize, stride=stride_len,
                                        padding=pad))
        self.conv_layers.append(nn.Conv2d(32, 32,
                                        kernel_size=kernelsize, stride=stride_len,
                                        padding=pad))
        self.conv_layers.append(nn.Conv2d(32, 64,
                                        kernel_size=kernelsize, stride=stride_len,
                                        padding=pad))
        self.pool = nn.MaxPool2d(kernel_size = kernelsize, stride = stride_len)
        self.flatten = nn.Flatten()
        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        ## Dense Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU(0.01))
            self.layers.append(nn.Dropout(dropout_rate) if dropout_rate else nn.Identity())
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
            x = self.pool(x)
            x = self.dropout(x)
        x = self.flatten(x)
        flatten_dim = x.shape[1]
        self.input_fc = nn.Linear(flatten_dim, self.hidden_dim)
        x = self.activation(self.input_fc(x))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_fc(x)
        return x