"""
PPO agent wrapper — supports both flat MaskablePPO and recurrent RecurrentPPO.

The key difference with the recurrent version:
  - lstm_states carries (h, c) tensors between steps within an episode
  - episode_start signals the LSTM to reset its hidden state
  - reset_episode() must be called at the start of each new game
"""

import numpy as np


class PPOAgent:
    def __init__(self, model):
        self._model = model
        self._lstm_states = None
        self._episode_start = np.array([True])
        self._recurrent = self._is_recurrent(model)

    @staticmethod
    def _is_recurrent(model) -> bool:
        return hasattr(model, "policy") and hasattr(model.policy, "lstm_actor")

    def reset_episode(self) -> None:
        """Call this at the start of every new game to wipe the LSTM state."""
        self._lstm_states = None
        self._episode_start = np.array([True])

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray,
                      board=None, deterministic: bool = False) -> int:
        obs_batch = obs[np.newaxis, ...]

        if self._recurrent:
            # Pass mask to the policy before prediction so it can apply it
            self._model.policy.set_action_mask(
                self._mask_tensor(action_mask)
            )
            action, self._lstm_states = self._model.predict(
                obs_batch,
                state=self._lstm_states,
                episode_start=self._episode_start,
                deterministic=deterministic,
            )
            self._episode_start = np.array([False])
        else:
            action, _ = self._model.predict(
                obs_batch,
                action_masks=action_mask[np.newaxis],
                deterministic=deterministic,
            )

        return int(action)

    def _mask_tensor(self, action_mask: np.ndarray):
        import torch
        return torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    def save(self, path: str) -> None:
        self._model.save(path)

    @classmethod
    def load(cls, path: str, recurrent: bool = True) -> "PPOAgent":
        if recurrent:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(path)
        else:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(path)
        return cls(model)
