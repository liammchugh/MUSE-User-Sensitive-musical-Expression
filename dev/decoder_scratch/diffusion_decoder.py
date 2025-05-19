"""
Trains a diffusion-based model to reconstruct/generate spectrograms
from latent codes. This is a simplified placeholder for an actual
diffusion pipeline (e.g. DDPM, DDIM).

May also adapt an open-source diffusion architecture
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LatentSpectrogramDataset(Dataset):
    """
    Dataset that loads (latent_vector, spectrogram) pairs.
    For demonstration, we generate random pairs or use a stored file.
    """
    def __init__(self, latent_file, spectrogram_file):
        self.latents = np.load(latent_file)  # shape (N, latent_dim)
        self.spectrograms = np.load(spectrogram_file)  # shape (N, freq, time)
        self.latents = self.latents.astype(np.float32)
        self.spectrograms = self.spectrograms.astype(np.float32)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.spectrograms[idx]

class SimpleDiffusionModel(nn.Module):
    """
    Very simplified placeholder for a diffusion U-Net.
    """
    def __init__(self, latent_dim=64, spec_shape=(128, 128)):
        super(SimpleDiffusionModel, self).__init__()
        in_features = latent_dim
        out_features = spec_shape[0] * spec_shape[1]

        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, out_features)

    def forward(self, latent):
        """
        A naive 'generation' approach: feed latent -> MLP -> reshape to spectrogram.
        Real diffusion would require time steps, noise schedules, etc.
        """
        x = torch.relu(self.fc1(latent))
        x = torch.sigmoid(self.fc2(x))
        return x

def diffusion_loss(predicted, target):
    return nn.functional.mse_loss(predicted, target, reduction='sum')

def train_diffusion(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (latent, spec) in enumerate(data_loader):
        latent = latent.to(device)
        spec = spec.to(device)
        batch_size = spec.size(0)
        # Flatten target spectrogram
        spec_flat = spec.view(batch_size, -1)

        optimizer.zero_grad()
        out = model(latent)
        loss = diffusion_loss(out, spec_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For a real pipeline, you'd create or load stored latent vectors 
    # by encoding the spectrograms through the trained VAE, 
    # then train the diffusion model to reconstruct the original spectrograms.

    # For demonstration, we assume random latents.
    # You can replace these with actual latents from the VAE's encoder.
    latent_file = "data/processed/vae_latents.npy"
    spectrogram_file = "data/processed/spectrograms.npy"
    if not os.path.exists(latent_file):
        # As a placeholder, we generate random latents:
        print("Generating random latent vectors as placeholder.")
        latent_dim = 64
        specs = np.load(spectrogram_file)
        num_samples = specs.shape[0]
        random_latents = np.random.randn(num_samples, latent_dim).astype(np.float32)
        np.save(latent_file, random_latents)

    dataset = LatentSpectrogramDataset(latent_file, spectrogram_file)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleDiffusionModel(latent_dim=64, spec_shape=(128, 128)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        train_loss = train_diffusion(data_loader, model, optimizer, device)
        print(f"Epoch {epoch}, Loss: {train_loss:.2f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/diffusion_decoder.pt")
    print("Diffusion decoder training complete and model saved.")

if __name__ == "__main__":
    main()
