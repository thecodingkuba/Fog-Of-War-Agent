"""
CNN feature extractor and recurrent, action-masked policy for Fog of War Chess.

Architecture
------------
  FogChessCNN              : Dict obs -> 256 features (CNN over the board planes)
  MaskedRecurrentPolicy    : CNN -> LSTM(256) -> actor / critic, with legal-move
                             masking applied to the action logits.

Why the mask lives in the observation
--------------------------------------
sb3-contrib provides RecurrentPPO and MaskablePPO separately but not a combined
masked + recurrent algorithm. Rather than smuggle the mask in through callbacks
(which desynchronizes rollout vs. gradient passes and silently breaks the PPO
ratio), we carry the action mask *inside the observation* as the "action_mask"
key. Because the rollout buffer stores observations, the exact same mask is
replayed during the gradient update, so sampling and log-prob computation stay
consistent.

The LSTM lets the agent integrate fogged observations over time and build an
internal belief state about hidden enemy pieces.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import (
    RecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FogChessCNN(BaseFeaturesExtractor):
    """
    Feature extractor for the Dict observation.

    Applies a small conv stack to the (14, 8, 8) "board" plane and ignores the
    "action_mask" key (the policy consumes the mask directly).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        board_space = observation_space["board"]
        in_channels = board_space.shape[0]  # 14

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
            dummy = torch.zeros(1, *board_space.shape)
            flat_dim = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(self.cnn(observations["board"]))


class MaskedRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    """RecurrentMultiInputActorCriticPolicy + legal-move masking.

    Illegal logits are pushed to -1e8 before sampling, giving them ~0
    probability while keeping the gradient graph intact. The mask is read
    from the "action_mask" key of the observation on every forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_mask: torch.Tensor | None = None

    # -- mask plumbing ------------------------------------------------- #

    def set_action_mask(self, mask: torch.Tensor) -> None:
        self._action_mask = mask

    @staticmethod
    def _mask_from_obs(obs) -> torch.Tensor | None:
        if isinstance(obs, dict) and "action_mask" in obs:
            return obs["action_mask"] > 0.5
        return None

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        dist = super()._get_action_dist_from_latent(latent_pi)
        if self._action_mask is not None:
            logits = dist.distribution.logits
            masked = logits.masked_fill(~self._action_mask, -1e8)
            dist.distribution = torch.distributions.Categorical(logits=masked)
        return dist

    # -- entry points: set the mask, then defer to the base policy ----- #

    def forward(self, obs, lstm_states, episode_starts, deterministic: bool = False):
        self._action_mask = self._mask_from_obs(obs)
        return super().forward(obs, lstm_states, episode_starts, deterministic)

    def get_distribution(self, obs, lstm_states, episode_starts):
        self._action_mask = self._mask_from_obs(obs)
        return super().get_distribution(obs, lstm_states, episode_starts)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        self._action_mask = self._mask_from_obs(obs)
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts)

    def predict_values(self, obs, lstm_states, episode_starts):
        # Value head needs no mask, but extract_features still wants the dict.
        return super().predict_values(obs, lstm_states, episode_starts)

    # -- helper for the search / prior agents -------------------------- #

    def get_action_scores(
        self,
        obs: dict[str, np.ndarray],
        lstm_state: tuple | None,
        episode_start: np.ndarray,
    ) -> tuple[np.ndarray, tuple]:
        """
        Return masked action logits and advance the actor LSTM state.

        obs:         single (unbatched) dict with numpy "board" and
                     "action_mask" arrays.
        lstm_state:  (h, c) numpy arrays of shape (n_layers, 1, hidden), or None.
        Returns:     (logits shape (4096,), new_lstm_state in the same format).
        """
        obs_t = {
            k: torch.as_tensor(v[np.newaxis], dtype=torch.float32, device=self.device)
            for k, v in obs.items()
        }
        ep_t = torch.as_tensor(episode_start, dtype=torch.float32, device=self.device)

        if lstm_state is None:
            shape = (self.lstm_actor.num_layers, 1, self.lstm_actor.hidden_size)
            h = torch.zeros(shape, device=self.device)
            c = torch.zeros(shape, device=self.device)
        else:
            h = torch.as_tensor(lstm_state[0], dtype=torch.float32, device=self.device)
            c = torch.as_tensor(lstm_state[1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist, new_state = self.get_distribution(obs_t, (h, c), ep_t)
            logits = dist.distribution.logits.cpu().numpy()[0]

        new_h, new_c = new_state
        return logits, (new_h.cpu().numpy(), new_c.cpu().numpy())
