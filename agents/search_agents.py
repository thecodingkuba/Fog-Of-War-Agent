"""
Action-prior agents: score legal moves with a prior, take top-k, run
restricted minimax over the candidates.

Paper baselines:
  HeuristicPriorAgent   — MVV-LVA captures + check bonus
  PolicyPriorAgent      — PPO action distribution (AlphaZero-style)
  InformationPriorAgent — PPO scores + FoW information gain (Ours)

All agents expose:
  select_action(obs, action_mask, board) → int
  reset_episode()
  total_nodes: int   (cumulative search nodes over the episode)
"""

import numpy as np
import chess

from env.board_utils import PIECE_VALUES, move_to_action
from search.information_gain import information_gain_scores
from search.minimax import restricted_search

_CAPTURE_W = 1.0
_CHECK_W = 0.3
_INFO_W = 0.05   # info gain: 0-30 squares → adds up to ~1.5 on top of logits


def _heuristic_scores(board: chess.Board) -> np.ndarray:
    scores = np.zeros(4096, dtype=np.float32)
    for move in board.legal_moves:
        a = move_to_action(move)
        victim = board.piece_at(move.to_square)
        if victim:
            scores[a] += PIECE_VALUES.get(victim.piece_type, 0) * _CAPTURE_W
        if board.is_en_passant(move):
            scores[a] += PIECE_VALUES[chess.PAWN] * _CAPTURE_W
        board.push(move)
        if board.is_check():
            scores[a] += _CHECK_W
        board.pop()
    return scores


def _top_k(scores: np.ndarray, action_mask: np.ndarray, k: int) -> np.ndarray:
    legal = np.where(action_mask)[0]
    if len(legal) <= k:
        return legal
    top_idx = np.argpartition(scores[legal], -k)[-k:]
    return legal[top_idx]


class HeuristicPriorAgent:
    """Hand-designed prior (MVV-LVA + check) + restricted minimax search."""

    def __init__(self, k: int = 10, depth: int = 2):
        self.k = k
        self.depth = depth
        self.total_nodes = 0

    def reset_episode(self) -> None:
        self.total_nodes = 0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray,
                      board: chess.Board | None = None) -> int:
        if board is None:
            raise ValueError("HeuristicPriorAgent requires board access")
        scores = _heuristic_scores(board)
        scores[action_mask] += 1e-3
        candidates = _top_k(scores, action_mask, self.k)
        if self.depth == 0:
            return int(candidates[np.argmax(scores[candidates])])
        action, nodes = restricted_search(board, candidates, self.depth)
        self.total_nodes += nodes
        return action


class PolicyPriorAgent:
    """
    PPO policy distribution as prior + restricted search (AlphaZero-style).

    Uses get_action_scores() on MaskedRecurrentPolicy to obtain the full
    masked action logit vector in one forward pass — no sampling approximation.
    The actor LSTM state is advanced via the same call.
    """

    def __init__(self, model, k: int = 10, depth: int = 2):
        self._model = model
        self.k = k
        self.depth = depth
        self._lstm_state = None
        self._episode_start = np.array([True])
        self.total_nodes = 0

    def reset_episode(self) -> None:
        self._lstm_state = None
        self._episode_start = np.array([True])
        self.total_nodes = 0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray,
                      board: chess.Board | None = None) -> int:
        if board is None:
            raise ValueError("PolicyPriorAgent requires board access")

        logits, self._lstm_state = self._model.policy.get_action_scores(
            obs[np.newaxis], self._lstm_state, self._episode_start, action_mask
        )
        self._episode_start = np.array([False])

        candidates = _top_k(logits, action_mask, self.k)
        if self.depth == 0:
            return int(candidates[np.argmax(logits[candidates])])
        action, nodes = restricted_search(board, candidates, self.depth)
        self.total_nodes += nodes
        return action

    @classmethod
    def load(cls, path: str, k: int = 10, depth: int = 2) -> "PolicyPriorAgent":
        from sb3_contrib import RecurrentPPO
        return cls(RecurrentPPO.load(path), k=k, depth=depth)


class InformationPriorAgent:
    """
    PPO scores augmented with FoW information gain + restricted search (Ours).

    Prior = π_prior(a|x_t) + info_weight × InfoGain(a)
    Matches paper diagram: FoW Information Signal feeds into Top-k selection
    alongside the Action Prior Network output.
    """

    def __init__(self, model, k: int = 10, depth: int = 2,
                 info_weight: float = _INFO_W):
        self._model = model
        self.k = k
        self.depth = depth
        self.info_weight = info_weight
        self._lstm_state = None
        self._episode_start = np.array([True])
        self.total_nodes = 0

    def reset_episode(self) -> None:
        self._lstm_state = None
        self._episode_start = np.array([True])
        self.total_nodes = 0

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray,
                      board: chess.Board | None = None) -> int:
        if board is None:
            raise ValueError("InformationPriorAgent requires board access")

        logits, self._lstm_state = self._model.policy.get_action_scores(
            obs[np.newaxis], self._lstm_state, self._episode_start, action_mask
        )
        self._episode_start = np.array([False])

        info = information_gain_scores(board)
        scores = logits + self.info_weight * info

        candidates = _top_k(scores, action_mask, self.k)
        if self.depth == 0:
            return int(candidates[np.argmax(scores[candidates])])
        action, nodes = restricted_search(board, candidates, self.depth)
        self.total_nodes += nodes
        return action

    @classmethod
    def load(cls, path: str, k: int = 10, depth: int = 2,
             info_weight: float = _INFO_W) -> "InformationPriorAgent":
        from sb3_contrib import RecurrentPPO
        return cls(RecurrentPPO.load(path), k=k, depth=depth, info_weight=info_weight)
