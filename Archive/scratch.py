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

epochs = 10
num_classes = 2
num_features = len(junco_data.columns) - 3
folds = 5
results = {}
batchsize = 10
kfold = model_selection.KFold(n_splits = folds, shuffle = True, random_state = 6260)
for params in itertools.product(*parameter_grid.values()):
    num_layers, dropout_rate, learning_rate, num_conv, kernelsize, pad, stride_len, hidden_dim = params
    scores = []
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        trainloader = DataLoader(train_subset, batch_size = batchsize, shuffle = True)
        testloader = DataLoader(test_subset, batch_size = batchsize)

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