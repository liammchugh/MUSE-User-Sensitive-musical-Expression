#!/usr/bin/env python3
"""
group_ppo_rlhf.py

RLHF fine-tuning base script using a PPO-like algorithm
with a group-relative baseline. The "policy" outputs a distribution
over latents given accelerometer data. A decoder (e.g., diffusion)
then maps that latent to a music embedding. The reward model scores
the embedding. PPO update is performed, with advantage computed
relative to the group's mean reward.

High-Level Steps:
1. Load pre-trained accelerometer->latent (policy) and decoder models.
2. Wrap them in a new "Policy" class that outputs (latent_dist, log_prob).
3. Sample multiple latents per accelerometer input to form a group.
4. Pass latents to decoder -> get embeddings -> compute reward.
5. Compute advantage = reward_i - mean(reward_in_group).
6. Perform standard PPO steps using the ratio of new vs. old log_probs, clipped objective, etc.
7. Update policy (accelerometer->latent) and possibly the decoder or keep it partially frozen.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Example "Decoder" - you might import from your own code
# e.g., from cognitive_music.decoder.diffusion_decoder import SimpleDiffusionModel
# For demonstration, define a stub decoder below:
# -------------------------------------------------------------------------
class StubDecoder(nn.Module):
    """ Simplified 'decoder' that turns a latent into an 'audio embedding'. """
    def __init__(self, latent_dim=64, output_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        """
        z: shape (batch_size, latent_dim)
        returns: shape (batch_size, output_dim) as 'music embedding'
        """
        return self.fc(z)

# -------------------------------------------------------------------------
# Example "RewardModel" - you might import from your own code
# e.g., from your RLHF pipeline
# -------------------------------------------------------------------------
class RewardModel(nn.Module):
    """
    Takes an audio embedding, outputs a scalar reward.
    Can incorporate group structure, but here we just do per-sample scoring.
    """
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embedding):
        """
        embedding: (batch_size, embedding_dim)
        Returns: (batch_size, 1) reward
        """
        return self.net(embedding)

# -------------------------------------------------------------------------
# PPO Policy: Accelerometer->Latent distribution + log_prob
# We treat the latent as a continuous action from a diagonal Gaussian.
# The decoder is typically part of the environment or the policy head,
# depending on whether we want to backprop through it. Here, we keep it
# in the policy so we can fine-tune it as well.
# -------------------------------------------------------------------------
class LatentPolicy(nn.Module):
    """
    Combines:
    1. A small network that outputs mean + log_std for latent from accelerometer input.
    2. A decoder that produces final audio embedding from latent.

    The "action" is the latent vector. The final "state" we pass to
    the reward model is the output of the decoder.
    """
    def __init__(self, accel_dim=3, latent_dim=64, hidden_dim=128, decoder=None):
        super().__init__()
        self.latent_dim = latent_dim
        # MLP from accel -> (mean, log_std) of latent
        self.fc = nn.Sequential(
            nn.Linear(accel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean + log_std
        )
        self.decoder = decoder  # e.g., a diffusion or simpler FC

    def forward(self, accel_features):
        """
        accel_features: (batch_size, accel_dim)
        returns:
         - distribution parameters (means, log_stds)
         - final embedding (if you want a 'deterministic' forward)
        Typically in PPO, we separate sampling from forward for clarity.
        """
        # (batch_size, 2*latent_dim)
        out = self.fc(accel_features)
        mean, log_std = out.split(self.latent_dim, dim=1)
        return mean, log_std

    def decode_latent(self, z):
        """
        Pass latent z to decoder -> audio embedding
        """
        return self.decoder(z)

# -------------------------------------------------------------------------
# Helper: Diagonal Gaussian sampling and log_prob
# -------------------------------------------------------------------------
def sample_latent_and_log_prob(mean, log_std):
    """
    Given mean, log_std (batch_size, latent_dim),
    sample z ~ Normal(mean, std).
    Return z, log_prob (elementwise).
    """
    std = log_std.exp()
    eps = torch.randn_like(std)
    z = mean + std * eps

    # log_prob under diagonal Gaussian
    # shape: (batch_size,)
    # Summation across latent_dim
    var = std * std
    log_prob = -0.5 * ((z - mean)**2 / var + 2*log_std + np.log(2*np.pi)).sum(dim=1)
    return z, log_prob

def compute_log_prob(mean, log_std, z):
    """
    Compute log_prob of z under N(mean, diag(std^2)).
    Used for PPO old vs new policy ratio.
    """
    std = log_std.exp()
    var = std * std
    log_p = -0.5 * ((z - mean)**2 / var + 2*log_std + np.log(2*np.pi))
    return log_p.sum(dim=1)  # sum across latent_dim

# -------------------------------------------------------------------------
# PPO Trainer: orchestrates data collection, advantage calculation,
# and PPO update steps with group-relative baseline.
# -------------------------------------------------------------------------
class GroupPPOTrainer:
    def __init__(
        self,
        policy,
        reward_model,
        device,
        ppo_epochs=5,
        batch_size=16,
        group_size=4,
        clip_range=0.2,
        lr=1e-4
    ):
        """
        policy: LatentPolicy
        reward_model: RewardModel
        device: torch.device
        ppo_epochs: how many epochs per PPO update
        batch_size: how many "groups" per iteration
        group_size: how many samples per group
        clip_range: PPO clipping epsilon
        lr: learning rate
        """
        self.policy = policy
        self.reward_model = reward_model
        self.device = device
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.clip_range = clip_range

        # We'll keep an "old policy" around for PPO ratio
        self.old_policy = copy.deepcopy(self.policy).eval()

        # Optimizer for policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # Typically the reward model is fixed or updated from actual labels
        # If you'd like to fine-tune it with new labels, add a separate optimizer or process.

    def sample_batch(self, accel_data):
        """
        For demonstration, we do the following:
        1) For each "group" in the batch, pick an accelerometer state from accel_data.
        2) Sample group_size latents from the policy distribution.
        3) Compute reward for each sample.
        4) Store (state, latent, reward, log_prob) for PPO.

        Returns lists (or concatenated Tensors) to feed into the PPO update.
        """
        # We'll assume accel_data is shape (N, 3).
        # We randomly pick 'batch_size' states from it.
        N = accel_data.shape[0]
        idx = np.random.choice(N, self.batch_size, replace=True)
        chosen_states = accel_data[idx].to(self.device)  # (batch_size, accel_dim)

        all_states = []
        all_latents = []
        all_rewards = []
        all_log_probs = []
        all_old_log_probs = []

        # We'll process each group separately
        for i in range(self.batch_size):
            state_i = chosen_states[i].unsqueeze(0)  # shape (1, accel_dim)

            # Evaluate policy distribution
            with torch.no_grad():
                mean_old, log_std_old = self.old_policy(state_i)
            mean_new, log_std_new = self.policy(state_i)

            # Sample group_size latents from the new policy
            latents_i = []
            log_probs_i = []
            old_log_probs_i = []
            rewards_i = []

            for _ in range(self.group_size):
                z, lp = sample_latent_and_log_prob(mean_new, log_std_new)
                latents_i.append(z)
                log_probs_i.append(lp)

                # Old policy log prob
                with torch.no_grad():
                    lp_old = compute_log_prob(mean_old, log_std_old, z)
                old_log_probs_i.append(lp_old)

            # Stack group results: (group_size, latent_dim), (group_size,)
            latents_i = torch.cat(latents_i, dim=0)  # each sample -> (batch=1, latent_dim) => stack => (group_size, latent_dim)
            log_probs_i = torch.cat(log_probs_i, dim=0)  # (group_size,)
            old_log_probs_i = torch.cat(old_log_probs_i, dim=0)

            # Decode to audio embeddings
            embeddings_i = self.policy.decode_latent(latents_i)  # (group_size, embedding_dim)
            # Compute rewards
            with torch.no_grad():
                # shape (group_size, 1)
                reward_i = self.reward_model(embeddings_i).squeeze(-1)  # -> (group_size,)

            latents_i = latents_i.detach()
            log_probs_i = log_probs_i.detach()
            old_log_probs_i = old_log_probs_i.detach()
            reward_i = reward_i.detach()

            # Accumulate
            # We'll replicate 'state_i' group_size times so we can keep them aligned
            # shape (group_size, accel_dim)
            states_i = state_i.repeat(self.group_size, 1)

            all_states.append(states_i)
            all_latents.append(latents_i)
            all_rewards.append(reward_i)
            all_log_probs.append(log_probs_i)
            all_old_log_probs.append(old_log_probs_i)

        # Concatenate across all groups
        states_tensor = torch.cat(all_states, dim=0)
        latents_tensor = torch.cat(all_latents, dim=0)
        rewards_tensor = torch.cat(all_rewards, dim=0)
        log_probs_tensor = torch.cat(all_log_probs, dim=0)
        old_log_probs_tensor = torch.cat(all_old_log_probs, dim=0)

        return states_tensor, latents_tensor, rewards_tensor, log_probs_tensor, old_log_probs_tensor

    def ppo_update(self, states, latents, rewards, old_log_probs, clip_range=0.2):
        """
        Standard PPO update step:
        1) Forward states with current policy to get distribution.
        2) Compute new log_prob of latents.
        3) Compute advantage. Here we do group-relative advantage:
            advantage_i = reward_i - mean(reward_of_that_group).
        4) Compute ratio = exp(new_log_prob - old_log_prob).
        5) Weighted clipped loss: L = min(ratio * advantage, clip(ratio, 1-clip, 1+clip)*advantage).
        """
        # Number of total samples = batch_size * group_size
        # We can recover group indexing if we want to compute group baselines per group.
        # For simplicity, let's do a simpler approach:
        # "Group" was each block of 'group_size' samples, so we can do it in lumps.
        # If the data is [B*g, ...], let's reshape to [B, g, ...] for advantage computation.
        B = self.batch_size
        g = self.group_size
        # shape => (B, g)
        rewards_2d = rewards.view(B, g)

        # Group baseline = mean reward in each group
        group_means = rewards_2d.mean(dim=1, keepdim=True)  # (B, 1)
        advantages_2d = rewards_2d - group_means  # shape (B, g)
        advantages = advantages_2d.view(-1)  # flatten back: (B*g,)

        # We'll forward pass again to get new log_probs
        with torch.set_grad_enabled(True):
            mean, log_std = self.policy(states)  # (B*g, latent_dim) each
            new_log_probs = compute_log_prob(mean, log_std, latents)

        # ratio = exp(new_log_prob - old_log_prob)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO clip objective
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2))  # negative because we do gradient ascent

        return policy_loss

    def train_one_iteration(self, accel_data):
        """
        1) Sample a batch of states from accel_data, each with group_size draws from policy.
        2) For multiple epochs, run PPO update steps.
        """
        # Freeze old policy
        self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        self.old_policy.eval()

        # Collect rollouts
        states, latents, rewards, log_probs, old_log_probs = self.sample_batch(accel_data)

        # PPO update
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            loss = self.ppo_update(
                states=states,
                latents=latents,
                rewards=rewards,
                old_log_probs=old_log_probs,
                clip_range=self.clip_range
            )
            loss.backward()
            self.optimizer.step()

        return loss.item(), rewards.mean().item()

# -------------------------------------------------------------------------
# MAIN: Example usage
# -------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate policy
    # Suppose your existing "Accelerometer->Latent" model was an MLP with output_dim=latent_dim
    # For PPO, we replace it with a distribution head that outputs mean+log_std
    # but we can initialize from your pre-trained weights, if desired.

    # Create a stub decoder or your actual diffusion model
    decoder = StubDecoder(latent_dim=64, output_dim=64).to(device)

    # Full policy
    policy = LatentPolicy(
        accel_dim=3,    # 3-axis accelerometer
        latent_dim=64,  # dimension of latent
        hidden_dim=128,
        decoder=decoder
    ).to(device)

    # 2. Instantiate a reward model
    reward_model = RewardModel(embedding_dim=64).to(device)

    # 3. Create a PPO trainer
    trainer = GroupPPOTrainer(
        policy=policy,
        reward_model=reward_model,
        device=device,
        ppo_epochs=3,     # how many updates per iteration
        batch_size=8,     # how many groups per iteration
        group_size=4,     # how many samples per group
        clip_range=0.2,
        lr=1e-4
    )

    # 4. Dummy dataset for accelerometer states
    # In real usage, load from your dataset or from streaming sensor data
    dummy_accel_data = torch.rand(100, 3)  # 100 samples, each 3-axis

    # 5. Train loop
    num_iterations = 10
    for iter_i in range(num_iterations):
        loss_val, avg_reward = trainer.train_one_iteration(dummy_accel_data)
        print(f"Iter {iter_i}/{num_iterations} | Loss: {loss_val:.4f} | Avg Reward: {avg_reward:.4f}")

    # 6. Save the updated policy
    os.makedirs("models", exist_ok=True)
    torch.save(trainer.policy.state_dict(), "models/acc_decoder_ppo.pt")
    print("Saved updated policy model to models/acc_decoder_ppo.pt")

if __name__ == "__main__":
    main()
