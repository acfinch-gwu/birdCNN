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
import bird_data as data
class JuncoDataset(Dataset):
    def __init__(self, tensor_data, labels):
        self.tensor_data = tensor_data
        self.labels = labels
        self.C, self.H, self.W = tensor_data.shape

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        return self.tensor_data[idx], self.labels[idx]

    def get_input_dim(self):
        return self.C

    def get_true_labels(self):
        return self.labels


class JuncoDatasetBuilder:
    def __init__(self, dataframe, feature_cols=None, label_col='species_observed',
                 num_lat_bins=100, num_lon_bins=100, fill_missing=True, fill_method='zeros'):
        self.df = dataframe.copy()
        self.label_col = label_col
        self.num_lat_bins = num_lat_bins
        self.num_lon_bins = num_lon_bins
        self.fill_missing = fill_missing
        self.fill_method = fill_method

        if feature_cols is None:
            exclude = ['latitude', 'longitude', 'lat_bin', 'lon_bin', label_col]
            self.feature_cols = [col for col in self.df.columns if col not in exclude]
        else:
            self.feature_cols = feature_cols

        self.grid = None
        self.labels = None

    def bin_coordinates(self):
        lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
        lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()

        lat_bins = np.linspace(lat_min, lat_max, self.num_lat_bins + 1)
        lon_bins = np.linspace(lon_min, lon_max, self.num_lon_bins + 1)

        self.df['lat_bin'] = np.digitize(self.df['latitude'], lat_bins) - 1
        self.df['lon_bin'] = np.digitize(self.df['longitude'], lon_bins) - 1

        self.df['lat_bin'] = self.df['lat_bin'].clip(0, self.num_lat_bins - 1)
        self.df['lon_bin'] = self.df['lon_bin'].clip(0, self.num_lon_bins - 1)

    def build_grid(self):
        self.bin_coordinates()
        grid = np.zeros((self.num_lat_bins, self.num_lon_bins, len(self.feature_cols)), dtype=np.float32)
        counts = np.zeros((self.num_lat_bins, self.num_lon_bins), dtype=np.int32)

        for _, row in self.df.iterrows():
            i, j = row['lat_bin'], row['lon_bin']
            grid[i, j] += row[self.feature_cols].values.astype(np.float32)
            counts[i, j] += 1

        nonzero_mask = counts > 0
        grid[nonzero_mask] /= counts[nonzero_mask, None]

        if self.fill_missing:
            if self.fill_method == 'mean':
                global_mean = np.nanmean(grid[nonzero_mask], axis=0)
                grid[~nonzero_mask] = global_mean
            if self.fill_method == 'zeros':
                grid[~nonzero_mask] = 0

        self.grid = grid

    def extract_labels(self):
        labels = np.zeros((self.num_lat_bins, self.num_lon_bins), dtype=np.int64)
        for _, row in self.df.iterrows():
            i, j = row['lat_bin'], row['lon_bin']
            labels[i, j] = row[self.label_col]
        self.labels = labels

    def get_dataset(self):
        if self.grid is None:
            self.build_grid()
        if self.labels is None:
            self.extract_labels()

        tensor_data = torch.from_numpy(self.grid).permute(2, 0, 1)
        labels = torch.tensor(self.labels)
        return JuncoDataset(tensor_data, labels)

builder = JuncoDatasetBuilder(data.junco_data, label_col='species_observed',
                              num_lat_bins = 100, num_lon_bins = 100)
dataset = builder.get_dataset()
length = len(dataset)
print(length)
input_dim = dataset.get_input_dim()
print(input_dim)
truth = dataset.get_true_labels()
truth = pd.DataFrame(truth.numpy())
