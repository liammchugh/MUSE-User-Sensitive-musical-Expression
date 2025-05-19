"""
Generate dummy token and VAE embedding data for demonstration.
Replace with real data code in practice.
"""

import numpy as np
import os

def create_dummy_data(num_sequences=1000, seq_len=130, vocab_size=256, latent_dim=128):
    """
    Generate random tokens and random VAE embeddings.
    We'll store them as .npy files in a 'dummy_data' folder.
    """
    os.makedirs("dummy_data", exist_ok=True)

    tokens = np.random.randint(low=0, high=vocab_size, size=(num_sequences, seq_len))
    # Example: shape (num_sequences, latent_dim)
    vae_embeddings = np.random.randn(num_sequences, latent_dim).astype(np.float32)

    np.save("dummy_data/tokens.npy", tokens)
    np.save("dummy_data/vae_embeddings.npy", vae_embeddings)
    print("Dummy data created in dummy_data/ folder.")

if __name__ == "__main__":
    create_dummy_data()
