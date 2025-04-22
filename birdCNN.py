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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parameter_grid = {
    'num_layers': [2, 3, 4, 5, 6], 
    'dropout_rate': [0, 0.1, 0.2], 
    'learning_rate': [0.01, 0.005, 0.001],
    'num_conv': [2, 4, 6], 
    'kernelsize': [2, 3, 4, 5], 
    'pad': [0, 1, 2], 
    'stride_len': [1, 2],
    'hidden_dim': [64, 128, 256, 512, 1024],
}

junco_data = data.junco_data.drop(columns = 'observation_date').dropna()

class JuncoDataset(Dataset):
    def __init__(self, tensor_data, labels):
        self.tensor_data = tensor_data
        self.labels = labels
        self.C, self.H, self.W = tensor_data.shape

        self.features = tensor_data.permute(1, 2, 0).reshape(-1, self.C)
        self.targets = labels.flatten()

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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
            grid[i, j] = row[self.feature_cols].values.astype(np.float32)
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

builder = JuncoDatasetBuilder(junco_data, label_col='species_observed',
                              num_lat_bins = 100, num_lon_bins = 100, fill_method = 'mean')
dataset = builder.get_dataset()

# def infer_flattened_size(model, input_shape):
#     with torch.no_grad():
#         dummy = torch.zeros(1, *input_shape).to(device)
#         out = model(dummy)
#         return out.view(1, -1).shape[1]

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
        self.input_fc = nn.Linear(flatten_dim, self.hidden_dim).to(device)
        x = self.activation(self.input_fc(x))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_fc(x)
        return x
    
epochs = 10
num_classes = 2
num_features = dataset.get_input_dim()
folds = 5
results = {}
batchsize = 32
kfold = model_selection.KFold(n_splits = folds, shuffle = True, random_state = 6260)
for params in itertools.product(*parameter_grid.values()):
    num_layers, dropout_rate, learning_rate, kernelsize, pad, stride_len, hidden_dim = params
    scores = []
    for folds, (train_index, test_index) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        trainloader = DataLoader(train_subset, batch_size = batchsize, shuffle = True)
        testloader = DataLoader(test_subset, batch_size = batchsize)

        model = birdNN(input_dim = num_features, output_dim = num_classes,
                       num_layers = num_layers, dropout_rate = dropout_rate,
                       kernelsize = kernelsize, pad = pad, stride_len = stride_len, 
                       hidden_dim = hidden_dim
                )
        criterion = nn.CrossEntropyLoss().to(device)
        accuracy = metrics.BinaryAccuracy().to(device)
        auroc = metrics.BinaryAUROC().to(device)
        # confusion_matrix = metrics.functional.multiclass_confusion_matrix
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma  = 0.75)

        print("Starting Model Training")
        model.train()
        train_loss = []
        train_accuracy = []
        train_auroc = []

        test_accuracy = []
        test_auroc = []

        for epoch in range(epochs):
            ## Model Training
            running_loss = 0
            for inputs, presence in trainloader:
                inputs, presence = inputs.to(device), presence.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, presence)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                accuracy.update(outputs, presence)
                auroc.update(outputs, presence)
            train_accuracy.append(accuracy.compute().numpy())
            train_auroc.append(auroc.compute().numpy())
            train_loss.append(running_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}\n')
                print(f'Loss: {train_loss[epoch]}\n')
                print(f'Accuracy: {train_accuracy[epoch]}\n')
                print(f'AUROC: {train_auroc[epoch]}\n')

            ## Model Validation
            accuracy.reset()
            auroc.reset()
            model.eval()
            with torch.no_grad():
                for inputs, presence in testloader:
                    outputs = model(inputs)
                    accuracy.update(outputs, presence)
                    auroc.update(outputs, presence)
            
            test_accuracy.append(accuracy.compute().numpy())
            test_auroc.append(auroc.compute().numpy())

            if (epoch + 1) % 10 == 0:
                print(f'Test Accuracy: {test_accuracy[epoch]}\n')
                print(f'Test AUROC: {test_auroc[epoch]}\n')
            

            scores.append(accuracy)
            accuracy.reset()
            auroc.reset()
            model.train()

            ## Saves model state
            torch.save(model.state_dict(), f'./model_states/model_{folds}.pth')
    avg_score = np.mean(scores)
    results[params] = avg_score