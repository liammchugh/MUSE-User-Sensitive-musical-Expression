"""
Trains a Variational Autoencoder (VAE) on music spectrogram data
to learn a latent representation. Saves the model for later use.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_file):
        self.spectrograms = np.load(spectrogram_file)
        # Convert to float32 for PyTorch
        self.spectrograms = self.spectrograms.astype(np.float32)

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        # shape: (freq_bins, time_frames)
        spec = self.spectrograms[idx]
        return spec

class VAE(nn.Module):
    def __init__(self, input_shape=(128, 128), latent_dim=64):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        in_features = input_shape[0] * input_shape[1]

        # Encoder
        self.fc1 = nn.Linear(in_features, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, in_features)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE or BCE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def train_vae(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, spectrogram in enumerate(data_loader):
        spectrogram = spectrogram.to(device)
        # Flatten for linear layers
        spectrogram_flat = spectrogram.view(spectrogram.size(0), -1)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(spectrogram_flat)
        loss = vae_loss(recon_x, spectrogram_flat, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "data/processed/spectrograms.npy"
    
    # Hyperparameters
    batch_size = 32
    latent_dim = 64
    lr = 1e-3
    num_epochs = 5

    # Dataset and Loader
    dataset = SpectrogramDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and Optimizer
    input_shape = (128, 128)
    model = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, num_epochs+1):
        train_loss = train_vae(data_loader, model, optimizer, device)
        print(f"Epoch {epoch}, Loss: {train_loss:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vae_music.pt")
    print("VAE training complete and model saved.")

if __name__ == "__main__":
    main()
