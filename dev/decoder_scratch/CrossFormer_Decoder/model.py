#!/usr/bin/env python3
"""
model.py

Implements a cross-attention Transformer architecture for decoding:
- Takes input tokens (previously generated tokens) as the autoregressive input.
- Cross-attends to an embedding derived from VAE's latent space.

Uses PyTorch's built-in Transformer layers for convenience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELatentEncoder(nn.Module):
    """
    Transforms the single VAE embedding (or a sequence of embeddings)
    into 'encoder hidden states' for the Transformer cross-attention.
    For a single embedding, we can just replicate or project it 
    into a sequence if needed.
    """
    def __init__(self, latent_dim, transformer_d_model, num_latent_tokens=1):
        """
        Args:
            latent_dim (int): Dimension of VAE latent space.
            transformer_d_model (int): Dimensionality of the Transformer model.
            num_latent_tokens (int): We can replicate or learn embeddings 
                                     from a single latent to multiple "latent tokens".
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.transformer_d_model = transformer_d_model
        self.num_latent_tokens = num_latent_tokens

        # Project latent_dim -> d_model
        self.projection = nn.Linear(latent_dim, transformer_d_model)

        # Optional: If we want a small learned embedding for each "latent token"
        self.latent_token_embedding = nn.Parameter(
            torch.randn(num_latent_tokens, transformer_d_model)
        )

    def forward(self, vae_embedding):
        """
        vae_embedding: (batch_size, latent_dim)
        Return: (batch_size, num_latent_tokens, d_model)
        """
        bsz = vae_embedding.size(0)
        # project the single latent vector
        projected = self.projection(vae_embedding)  # (batch_size, d_model)

        # expand to (batch_size, num_latent_tokens, d_model)
        # each sample gets repeated or offset by a small learned embedding
        # Simple approach: add a learned "latent token embedding"
        # which is shape (num_latent_tokens, d_model).
        # We can either add or stack them.
        # Example: tile 'projected' across num_latent_tokens, then add the token embedding.
        expanded = projected.unsqueeze(1).expand(bsz, self.num_latent_tokens, self.transformer_d_model)
        # Add the learned embeddings
        expanded = expanded + self.latent_token_embedding.unsqueeze(0)

        return expanded


class CrossAttentionMusicTransformer(nn.Module):
    """
    A Transformer Decoder that cross-attends to VAELatentEncoder outputs.
    The 'encoder' states come from VAELatentEncoder (which replicates the single latent vector 
    or transforms it into a short sequence).
    The 'decoder' attends to input tokens (autoregressive) + cross-attends to the latent states.
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=512,
        latent_dim=128,
        num_latent_tokens=1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding (basic sinusoidal or learned)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # The "encoder" in cross-attention is just the latent embedding block
        self.latent_encoder = VAELatentEncoder(latent_dim, d_model, num_latent_tokens=num_latent_tokens)

        # Final linear layer to predict next token
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_tokens,
        vae_embedding,
        teacher_forcing=True
    ):
        """
        input_tokens: (batch_size, seq_len)
        vae_embedding: (batch_size, latent_dim)

        Return: (batch_size, seq_len, vocab_size)
        """
        bsz, seq_len = input_tokens.shape

        # 1. Encode tokens (embedding + positional)
        token_emb = self.token_embedding(input_tokens) * (self.d_model ** 0.5)
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        token_emb = token_emb + pos_emb  # shape: (batch_size, seq_len, d_model)

        # For the PyTorch transformer, we want shapes: (seq_len, batch_size, d_model)
        token_emb = token_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)

        # 2. Get "encoder" hidden states from the latent embedding
        # shape: (batch_size, num_latent_tokens, d_model)
        latent_states = self.latent_encoder(vae_embedding)

        # Transformer expects shape (S, N, E) for the source
        latent_states = latent_states.transpose(0, 1)  # (num_latent_tokens, batch_size, d_model)

        # 3. Create masks for autoregressive decoding
        # (We typically use a causal mask to prevent attending to future tokens)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_tokens.device)

        # 4. Decode
        # decoder output shape: (seq_len, batch_size, d_model)
        dec_output = self.transformer_decoder(
            tgt=token_emb,
            memory=latent_states,
            tgt_mask=tgt_mask
        )

        # 5. Output projection
        # shape: (seq_len, batch_size, vocab_size)
        logits = self.output_linear(dec_output)

        # rearrange to (batch_size, seq_len, vocab_size)
        logits = logits.transpose(0, 1)
        return logits

    def generate(
        self,
        start_tokens,
        vae_embedding,
        max_length=128,
        temperature=1.0
    ):
        """
        Autoregressive generation:
        - We'll decode one token at a time, feeding it back in.
        """
        self.eval()
        generated = [t for t in start_tokens]  # assume shape (seq_len,)
        device = start_tokens.device

        # Prepare latent states once
        with torch.no_grad():
            latent_states = self.latent_encoder(vae_embedding.unsqueeze(0))  # (1, num_latent_tokens, d_model)
            latent_states = latent_states.transpose(0, 1)  # (num_latent_tokens, 1, d_model)
        
        for _ in range(max_length - len(generated)):
            seq_len = len(generated)
            # We embed the entire sequence each time (inefficient for large seq, but straightforward)
            input_tokens = torch.tensor(generated, device=device).unsqueeze(0)  # (1, seq_len)
            token_emb = self.token_embedding(input_tokens) * (self.d_model ** 0.5)

            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
            token_emb = token_emb + pos_emb  # (1, seq_len, d_model)

            token_emb = token_emb.transpose(0, 1)  # (seq_len, 1, d_model)

            # causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            dec_output = self.transformer_decoder(
                tgt=token_emb,
                memory=latent_states,
                tgt_mask=tgt_mask
            )  # (seq_len, 1, d_model)

            logits = self.output_linear(dec_output[-1])  # last token logits, shape (1, vocab_size)
            logits = logits[0]  # shape (vocab_size,)

            # sample or take argmax
            if temperature == 1.0:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                # Softmax with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

        return torch.tensor(generated, device=device)
