import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import rasterio
import fiona

data_folder = "Pre-Processed Dark-Eyed Junco Data/"

checklist_zf = pd.read_csv(data_folder + 'checklists_zf_md_deju_jan.csv', index_col = 'checklist_id')
env_checklist = pd.read_csv(data_folder + 'environmental_vars_checklists_md_jan.csv', index_col = 'checklist_id')

env_prediction_grid = pd.read_csv(data_folder + 'environmental_vars_prediction_grid_md.csv')
layer_names = fiona.listlayers(data_folder + 'gis-data.gpkg')
gis_layers = {layer: gpd.read_file(data_folder + 'gis-data.gpkg', layer = layer) for layer in layer_names}
with rasterio.open(data_folder + 'prediction_grid_md.tif') as src:
    grid_array = src.read(1)
    prediction_grid = pd.DataFrame(grid_array)

parameter_grid = {
    'num_layers': [2, 3, 4], 
    'dropout_rate': [0, 0.1, 0.2], 
    # 'learning_rate': [0.01, 0.005, 0.001],
    'hidden_dim': [64, 128, 256],
    # 'batchsize': [32, 64, 128]
}

junco_data = pd.merge(checklist_zf, env_checklist, how = 'left', left_index = True, right_index = True)
junco_data.reset_index(inplace=True)
junco_data.drop(labels = ['checklist_id', 'observer_id', 'type', 'state_code', 'locality_id', 'protocol_type'], axis = 1, inplace = True)