"""
CNN feature extractor and recurrent LSTM policy for Fog of War Chess.

Architecture:
  FogChessCNN:        (14,8,8) → 256 features       [shared extractor]
  FogChessLSTMPolicy: CNN → LSTM(256) → actor/critic [recurrent policy]

The LSTM lets the agent integrate fog observations over time and build an
internal belief state about hidden enemy piece locations.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FogChessCNN(BaseFeaturesExtractor):
    """
    Input:  (batch, 14, 8, 8)
    Output: (batch, features_dim)
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]  # 14

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            flat_dim = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))


class MaskedRecurrentPolicy(RecurrentActorCriticCnnPolicy):
    """
    RecurrentActorCriticCnnPolicy + legal-move masking.

    sb3-contrib doesn't combine RecurrentPPO and MaskablePPO, so we apply
    the mask here: illegal logits are set to -1e8 before sampling, giving
    them ~0 probability without breaking the gradient graph.

    The action mask must be passed in the `action_masks` key of the info
    dict, which SB3 surfaces as obs_tensor when using ActionMasker wrapper —
    but here we read it from a thread-local set by SelfPlayEnv.step().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_mask: torch.Tensor | None = None

    def set_action_mask(self, mask: torch.Tensor) -> None:
        self._action_mask = mask

    def get_action_scores(
        self,
        obs: "np.ndarray",
        lstm_state: "tuple | None",
        episode_start: "np.ndarray",
        action_mask: "np.ndarray",
    ) -> "tuple[np.ndarray, tuple]":
        """
        Return masked action logits and advance the actor LSTM state.

        lstm_state: (h, c) numpy arrays shape (n_layers, 1, hidden_size), or None.
        Returns:    (logits shape (4096,), new_lstm_state same numpy format).
        """
        import numpy as np
        obs_t, _ = self.obs_to_tensor(obs)
        ep_t = torch.tensor(episode_start, dtype=torch.float32, device=self.device)

        if lstm_state is None:
            shape = (self.lstm_actor.num_layers, obs_t.shape[0], self.lstm_actor.hidden_size)
            h = torch.zeros(shape, device=self.device)
            c = torch.zeros(shape, device=self.device)
        else:
            h = torch.tensor(lstm_state[0], dtype=torch.float32, device=self.device)
            c = torch.tensor(lstm_state[1], dtype=torch.float32, device=self.device)

        self.set_action_mask(
            torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            dist, (new_h, new_c) = self.get_distribution(obs_t, (h, c), ep_t)
            logits = dist.distribution.logits.cpu().numpy()[0]

        return logits, (new_h.cpu().numpy(), new_c.cpu().numpy())

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        dist = super()._get_action_dist_from_latent(latent_pi)
        if self._action_mask is not None:
            # Apply mask: set logits for illegal actions to -1e8
            logits = dist.distribution.logits.clone()
            logits[~self._action_mask] = -1e8
            dist.distribution = torch.distributions.Categorical(logits=logits)
        return dist
