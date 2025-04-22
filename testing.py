import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import rasterio
import fiona
import sklearn.model_selection as model_selection
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torcheval.metrics as metrics

import matplotlib.pyplot as plt

data_folder = "Pre-Processed Dark-Eyed Junco Data/"

checklist_zf = pd.read_csv(data_folder + 'checklists_zf_md_deju_jan.csv', index_col = 'checklist_id', nrows = 1000)
env_checklist = pd.read_csv(data_folder + 'environmental_vars_checklists_md_jan.csv', index_col = 'checklist_id', nrows = 1000)
env_prediction_grid = pd.read_csv(data_folder + 'environmental_vars_prediction_grid_md.csv')
layer_names = fiona.listlayers(data_folder + 'gis-data.gpkg')
gis_layers = {layer: gpd.read_file(data_folder + 'gis-data.gpkg', layer = layer) for layer in layer_names}
with rasterio.open(data_folder + 'prediction_grid_md.tif') as src:
    grid_array = src.read(1)
    prediction_grid = pd.DataFrame(grid_array)

junco_data = pd.merge(checklist_zf, env_checklist, how = 'outer', left_index = True, right_index = True)
junco_data.reset_index(inplace=True)
junco_data.drop(labels = ['checklist_id', 'observer_id', 'type'], axis = 1, inplace = True)

parameter_grid = {
    'num_layers': [2, 3, 4, 5, 6], 
    'dropout_rate': [0, 0.1, 0.2], 
    'learning_rate': [0.01, 0.005, 0.001],
    'num_conv': [2, 4, 6], 
    'kernelsize': [2, 3, 4, 5], 
    'pad': [0, 1, 2], 
    'stride_len': [1, 2], 
    # 'input_dim': 2, 
    'hidden_dim': [64, 128, 256, 512, 1024], 
    # 'output_dim': 1, 
}

# Parameters
grid_size = (junco_data['latitude'].nunique(), junco_data['longitude'].nunique())  # height, width
features = junco_data.columns

# Get bounds
lat_min, lat_max = junco_data['latitude'].min(), junco_data['latitude'].max()
lon_min, lon_max = junco_data['longitude'].min(), junco_data['longitude'].max()

# Create bins
lat_bins = np.linspace(lat_min, lat_max, grid_size[0] + 1)
lon_bins = np.linspace(lon_min, lon_max, grid_size[1] + 1)

# Digitize coordinates into bins
junco_data['lat_bin'] = np.digitize(junco_data['latitude'], lat_bins) - 1
junco_data['lon_bin'] = np.digitize(junco_data['longitude'], lon_bins) - 1

# Clip to avoid out-of-bounds
junco_data['lat_bin'] = junco_data['lat_bin'].clip(0, grid_size[0] - 1)
junco_data['lon_bin'] = junco_data['lon_bin'].clip(0, grid_size[1] - 1)

# Initialize grid tensor
grid = np.zeros((grid_size[0], grid_size[1], len(features)))

# Count how many points in each cell for averaging
counts = np.zeros((grid_size[0], grid_size[1]))

# Fill the grid with mean feature values
for _, row in junco_data.iterrows():
    i, j = row['lat_bin'], row['lon_bin']
    grid[i, j] += np.array([row[f] for f in features])
    counts[i, j] += 1

# Avoid division by zero
nonzero_mask = counts > 0
grid[nonzero_mask] /= counts[nonzero_mask, None]

def conv_dim(input_dim, pad, kernelsize, stride_len):
    return ((input_dim + (2*pad) - kernelsize) / stride_len) + 1

def pool_dim(input_dim, kernelsize, stride_len):
    return ((input_dim - kernelsize) / stride_len) + 1

class birdNN(nn.Module):
    def __init__(self, num_layers, dropout_rate, 
                 num_conv, kernelsize, pad, stride_len, 
                 input_dim, hidden_dim, output_dim):
        super(birdNN, self).__init__()
        ## Convolution Layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(input_dim, 16, 
                                          kernel_size = kernelsize, stride = stride_len, 
                                          padding = pad))
        conv_size = conv_dim(input_dim, pad, kernelsize, stride_len)
        print(conv_size)
        for _ in range(num_conv):
            self.conv_layers.append(nn.Conv2d(16, 16, 
                                          kernel_size = kernelsize, stride = stride_len, 
                                          padding = pad))
            conv_size = conv_dim(conv_size, pad, kernelsize, stride_len)
            print(conv_size)
        self.conv_layers.append(nn.Conv2d(16, 32, 
                                          kernel_size = kernelsize, stride = stride_len, 
                                          padding = pad))
        conv_size = conv_dim(conv_size, pad, kernelsize, stride_len)
        self.pool = nn.MaxPool2d(kernel_size = kernelsize, stride = stride_len)
        pool_size = pool_dim(conv_size, kernelsize, stride_len)
        print(pool_size)
        self.flatten = nn.Flatten()
        self.activation = nn.LeakyReLU(0.01)
        if not dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        ## Dense Layers
        self.input_fc = nn.Linear(int(32 * pool_size * pool_size), hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU(0.01))
            if not dropout_rate:
                self.layers.append(nn.Dropout(dropout_rate))
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
            x = self.pool(x)
            x = self.dropout(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        x = self.activation(self.input_fc(x))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_fc(x)
        return F.softmax(x)
    
epochs = 10
num_classes = 2
num_features = junco_data.shape[1] - 1
folds = 5
results = {}
batchsize = 10
kfold = model_selection.KFold(n_splits = folds, shuffle = True)
for params in itertools.product(*parameter_grid.values()):
    num_layers, dropout_rate, learning_rate, num_conv, kernelsize, pad, stride_len, hidden_dim = params
    scores = []
    for folds, (train_index, test_index) in enumerate(kfold.split(junco_data)):
        train_subset = torch.utils.data.Subset(junco_data, train_index)
        test_subset = torch.utils.data.Subset(junco_data, test_index)
        
        trainloader = DataLoader(train_subset, batch_size = batchsize, shuffle = True)
        testloader = DataLoader(test_subset, batch_size = batchsize, shuffle = True)

        model = birdNN(input_dim = num_features, output_dim = num_classes,
                       num_layers = num_layers, dropout_rate = dropout_rate, 
                       num_conv = num_conv, kernelsize = kernelsize, pad = pad, stride_len = stride_len, 
                       hidden_dim = hidden_dim
                )
        criterion = nn.CrossEntropyLoss()
        accuracy = metrics.MulticlassAccuracy(num_classes = num_classes)
        auroc = metrics.MulticlassAUROC(num_classes = num_classes)
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
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, presence)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                accuracy.update(outputs, presence)
                auroc.update(outputs, presence)
            train_accuracy.append(accuracy.compute().numpy())
            train_auroc.append(accuracy.compute().numpy())
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
            torch.save(model.state_dict, f'./model_states/model_{folds}.pth')
    avg_score = np.mean(scores)
    results[params] = avg_score

