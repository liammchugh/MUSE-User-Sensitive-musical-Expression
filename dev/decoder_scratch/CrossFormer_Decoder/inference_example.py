"""
Loads the trained cross-attention Transformer and generates a token sequence
given VAE embedding.
"""

import torch
import numpy as np
from model import CrossAttentionMusicTransformer

def load_model(model_path, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, max_seq_len, latent_dim, num_latent_tokens):
    model = CrossAttentionMusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def main():
    # Load the model
    model = load_model(
        model_path="music_transformer.pt",
        vocab_size=256,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=512,
        latent_dim=128,
        num_latent_tokens=4
    )

    # Suppose we have a single VAE embedding from somewhere
    vae_embedding = torch.randn(128)  # shape (latent_dim,)

    # Start tokens
    start_tokens = torch.tensor([10, 15, 20])  # random example

    # Generate
    generated_seq = model.generate(
        start_tokens=start_tokens,
        vae_embedding=vae_embedding,
        max_length=64,
        temperature=1.0
    )

    print("Generated sequence:", generated_seq.tolist())

if __name__ == "__main__":
    main()
