"""
Coordinates:
1. Dataset loading
2. Model instantiation
3. Training loop (teacher forcing)
4. Saving the model

Usage Example:
    python train.py --tokens data/tokens.npy \
                    --vae-embeddings data/vae_embeddings.npy \
                    --vocab-size 256 --lr 1e-4
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MusicTokenDataset
from model import CrossAttentionMusicTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Train cross-attention Transformer for music tokens.")
    parser.add_argument("--tokens", type=str, required=True, help="Path to .npy file with token sequences.")
    parser.add_argument("--vae-embeddings", type=str, required=True, help="Path to .npy file with VAE embeddings.")
    parser.add_argument("--vocab-size", type=int, default=512, help="Vocabulary size for tokens.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimensionality of VAE latent.")
    parser.add_argument("--d-model", type=int, default=512, help="Transformer model dimension.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num-decoder-layers", type=int, default=6, help="Number of Transformer decoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Feedforward dimension in Transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--num-latent-tokens", type=int, default=4, help="Number of tokens to represent the VAE latent.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--save-path", type=str, default="music_transformer.pt", help="Path to save the trained model.")
    parser.add_argument("--seq-len", type=int, default=128, help="Truncate/segment sequence length for training data.")
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # or adapt if you have special padding
    total_loss = 0

    for input_tokens, target_tokens, vae_embedding in dataloader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        vae_embedding = vae_embedding.to(device)

        optimizer.zero_grad()
        logits = model(input_tokens, vae_embedding)  # (batch_size, seq_len, vocab_size)
        
        # Flatten for cross-entropy: (batch_size * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = logits.view(-1, vocab_size)
        targets_2d = target_tokens.view(-1)

        loss = criterion(logits_2d, targets_2d)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & Dataloader
    dataset = MusicTokenDataset(
        tokens_file=args.tokens,
        vae_embeddings_file=args.vae_embeddings,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Model
    model = CrossAttentionMusicTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        latent_dim=args.latent_dim,
        num_latent_tokens=args.num_latent_tokens
    ).to(device)

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")

    # 5. Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
