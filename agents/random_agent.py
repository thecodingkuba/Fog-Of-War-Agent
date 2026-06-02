"""Random agent — selects uniformly from legal moves each turn."""

import numpy as np


class RandomAgent:
    """Stateless agent that picks a random legal action given an action mask."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, board=None) -> int:
        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            raise RuntimeError("No legal actions available — environment is broken.")
        return int(self.rng.choice(legal))
