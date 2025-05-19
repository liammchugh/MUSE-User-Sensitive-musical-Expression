"""
Trains a model to map accelerometer features to the same latent space
used by the VAE (or to a 'conditioning vector' that influences generation).

We assume we have preprocessed accelerometer data in `accelerometer_features.npy`.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AccelerometerDataset(Dataset):
    def __init__(self, acc_file, label_file=None):
        self.acc_data = np.load(acc_file)  # shape (N, 3) or (N, <windowed_features>)
        self.acc_data = self.acc_data.astype(np.float32)
        self.labels = None
        if label_file is not None and os.path.exists(label_file):
            self.labels = np.load(label_file).astype(np.int64)

    def __len__(self):
        return len(self.acc_data)

    def __getitem__(self, idx):
        features = self.acc_data[idx]
        if self.labels is not None:
            return features, self.labels[idx]
        else:
            return features

class AccelerometerToLatent(nn.Module):
    """
    Maps accelerometer features to a latent code (same dimension as VAE latent_dim).
    """
    def __init__(self, input_dim=3, latent_dim=64):
        super(AccelerometerToLatent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_acc_model(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for batch_idx, acc_features in enumerate(data_loader):
        acc_features = acc_features.to(device)
        
        # For demonstration, we assume an auto-encoder style approach:
        # we try to map acc -> latent -> reconstruct acc (toy method).
        # In a real scenario, we might train this to match known latents from the VAE, 
        # or to predict certain mood/energy labels, etc.
        
        optimizer.zero_grad()
        predicted_latent = model(acc_features)
        # Fake "target" is zero for demonstration, you can replace with real latents
        target_latent = torch.zeros_like(predicted_latent).to(device)

        loss = criterion(predicted_latent, target_latent)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_file = "data/processed/accelerometer_features.npy"
    dataset = AccelerometerDataset(acc_file)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AccelerometerToLatent(input_dim=3, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        train_loss = train_acc_model(data_loader, model, optimizer, device)
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/acc_to_latent.pt")
    print("Accelerometer-to-latent model training complete.")

if __name__ == "__main__":
    main()
