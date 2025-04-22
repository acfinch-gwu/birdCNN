import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load data
data_folder = "Pre-Processed Dark-Eyed Junco Data/"
env = pd.read_csv(data_folder + "environmental_vars_checklists_md_jan.csv")
checklists = pd.read_csv(data_folder + "checklists_zf_md_deju_jan.csv")
train_df = pd.merge(checklists, env, on="checklist_id")

features = ['year', 'day_of_year', 'hours_of_day',
            'effort_hours', 'effort_distance_km', 'effort_speed_kmph',
            'number_observers'] + \
           [col for col in train_df.columns if col.startswith(('pland_', 'ed_', 'elevation_'))]

X = train_df[features]
y = train_df['species_observed'].astype(int)

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=28)

# 3. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Define the model
class SpeciesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SpeciesNet(X_train.shape[1]).to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# 6. Predict and calibrate with isotonic regression
model.eval()
with torch.no_grad():
    train_probs = model(X_train_tensor).cpu().numpy().flatten()

cal = IsotonicRegression(out_of_bounds='clip')
cal.fit(train_probs, y_train)

with torch.no_grad():
    val_probs_raw = model(X_val_tensor).cpu().numpy().flatten()
val_probs_cal = cal.predict(val_probs_raw)

# 7. Threshold tuning
best_mcc, best_f1, best_thresh = -1, -1, 0
for t in np.linspace(0, 1, 100):
    preds = (val_probs_cal > t).astype(int)
    m = matthews_corrcoef(y_val, preds)
    f = f1_score(y_val, preds)
    if m > best_mcc:
        best_mcc, best_f1, best_thresh = m, f, t

print(f"Best threshold: {best_thresh:.3f}, MCC: {best_mcc:.3f}, F1: {best_f1:.3f}")

# 8. Predict on grid
grid = pd.read_csv(data_folder + "environmental_vars_prediction_grid_md.csv")
grid["observation_date"] = pd.to_datetime("2023-01-15")
grid["year"] = grid["observation_date"].dt.year
grid["day_of_year"] = grid["observation_date"].dt.dayofyear
grid["hours_of_day"] = 7.5
grid["effort_distance_km"] = 2
grid["effort_hours"] = 1
grid["effort_speed_kmph"] = 2
grid["number_observers"] = 1

X_grid = grid[features]
X_grid_scaled = scaler.transform(X_grid)
X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    grid_probs_raw = model(X_grid_tensor).cpu().numpy().flatten()
grid_probs_cal = cal.predict(grid_probs_raw)
grid["encounter_rate"] = np.clip(grid_probs_cal, 0, 1)

# Save outputs
grid_output = grid[["cell_id", "x", "y", "encounter_rate"]]
grid_output["in_range"] = (grid_output["encounter_rate"] > best_thresh).astype(int)
grid_output.to_csv("junco_nn_predictions.csv", index=False)

# Save validation predictions for R
results_df = pd.DataFrame({
    'obs': y_val,
    'pred': val_probs_cal
})
results_df.to_csv("dnn_predictions_for_r.csv", index=False)
