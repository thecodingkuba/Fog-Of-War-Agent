"""
Fog of War Chess environment (Gymnasium API).

This is a *partially observable* chess environment built on python-chess. It
uses true Fog of War rules: there is no check or checkmate, so a player may
move into check, leave their king attacked, and the game is won by *capturing*
the enemy king. Moves are therefore pseudo-legal (king captures and moving into
check are allowed). Each player only observes the part of the board its own
pieces can see, which is what makes the task hard.

Observation (Dict)
-------------------
- "board":       float32 tensor of shape (14, 8, 8), from the perspective of
                 the side to move:
                   planes  0-5   own pieces      (P, N, B, R, Q, K)
                   planes  6-11  visible enemy   (P, N, B, R, Q, K)
                   plane   12    known-empty squares (visible & empty)
                   plane   13    fog (unknown squares + hidden enemy pieces)
- "action_mask": float32 vector of shape (4096,); 1 where an action is allowed.
                 With pruning enabled this is the top-k heuristic candidate set;
                 otherwise it is the full set of legal moves.

Action
------
Discrete(4096), encoded as from_square * 64 + to_square (see board_utils).

Reward
------
Returned from the perspective of the player that just moved:
  +1.0  that player captured the enemy king (win)
   0.0  draw / ongoing
  -1.0  the move was illegal (should not happen when the mask is respected)

The self-play wrapper in train.py negates rewards appropriately for the
opponent's turns.
"""

from __future__ import annotations

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.board_utils import (
    ACTION_SPACE_SIZE,
    PIECE_TO_PLANE,
    PIECE_VALUES,
    action_to_move,
    legal_action_mask,
    top_k_action_mask,
    visible_squares,
)

N_PLANES = 14
BOARD_SHAPE = (N_PLANES, 8, 8)

# Bonus (in "pawn" units) folded into material shaping when a move gives check.
CHECK_BONUS = 0.5


class FogChessEnv(gym.Env):
    """Single-agent view of Fog of War Chess; opponent is driven externally."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, prune_k: int | None = None, max_moves: int = 200,
                 shaping_coef: float = 0.0, track_info_gain: bool = False,
                 truncation_material_coef: float = 0.0):
        super().__init__()
        self.prune_k = prune_k
        self.max_moves = max_moves
        # Dense reward shaping: per-move reward = shaping_coef * (material the
        # mover gains from captures/promotions + a bonus for giving check).
        # 0 disables it (pure sparse win/loss reward). The self-play wrapper
        # negates the opponent's gains, so the learner is also penalized for
        # material it loses / checks it receives.
        self.shaping_coef = shaping_coef
        # When True, each step reports the fog-of-war information gain (number
        # of newly revealed squares) of the move just played in info["info_gain"].
        # Used by the self-play wrapper to train an information-seeking prior.
        self.track_info_gain = track_info_gain
        # When >0, a game that hits the move cap is no longer a flat draw: the
        # mover receives a (capped) reward proportional to its material balance,
        # so a material lead converts to a positive outcome instead of stalling
        # to 0. Capped below the +1.0 king-capture win so winning stays optimal.
        # The self-play wrapper's zero-sum negation propagates this to White
        # whether the truncating ply was White's or Black's.
        self.truncation_material_coef = truncation_material_coef

        self.board = chess.Board()
        self._move_count = 0

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0, high=1.0, shape=BOARD_SHAPE, dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(ACTION_SPACE_SIZE,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #

    def _action_mask(self) -> np.ndarray:
        """Allowed-action mask for the side to move (pruned if prune_k set)."""
        if self.prune_k is not None:
            return top_k_action_mask(self.board, self.prune_k)
        return legal_action_mask(self.board)

    def _board_tensor(self) -> np.ndarray:
        board = self.board
        color = board.turn
        obs = np.zeros(BOARD_SHAPE, dtype=np.float32)
        vis = visible_squares(board, color)

        for sq in chess.SQUARES:
            row, col = sq // 8, sq % 8
            piece = board.piece_at(sq)

            if piece is None:
                if vis[sq]:
                    obs[12, row, col] = 1.0   # known empty
                else:
                    obs[13, row, col] = 1.0   # fog
                continue

            if piece.color == color:
                obs[PIECE_TO_PLANE[piece.piece_type], row, col] = 1.0
            elif vis[sq]:
                obs[6 + PIECE_TO_PLANE[piece.piece_type], row, col] = 1.0
            else:
                obs[13, row, col] = 1.0       # hidden enemy piece -> fog
        return obs

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "board": self._board_tensor(),
            "action_mask": self._action_mask().astype(np.float32),
        }

    def _info(self) -> dict:
        return {"action_mask": self._action_mask()}

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = chess.Board()
        self._move_count = 0
        return self._get_obs(), self._info()

    def step(self, action: int):
        move = action_to_move(int(action), self.board)

        # Safeguard: an illegal action ends the game as a loss for the mover.
        if move not in self.board.pseudo_legal_moves:
            obs = self._get_obs()
            info = self._info()
            info["illegal_move"] = True
            return obs, -1.0, True, False, info

        mover = self.board.turn
        victim = self.board.piece_at(move.to_square)
        king_captured = victim is not None and victim.piece_type == chess.KING
        material = self._material_gain(move) if self.shaping_coef else 0.0
        vis_before = visible_squares(self.board, mover) if self.track_info_gain else None

        self.board.push(move)
        self._move_count += 1

        if (self.shaping_coef and not king_captured
                and self.board.king(self.board.turn) is not None
                and self.board.is_check()):
            material += CHECK_BONUS  # the move attacks the enemy king

        reward = self.shaping_coef * material
        terminated = False
        if king_captured:
            # Fog of War win condition: the mover captured the enemy king.
            terminated = True
            reward += 1.0
        elif self._is_drawn():
            terminated = True

        truncated = self._move_count >= self.max_moves and not terminated
        if truncated and self.truncation_material_coef:
            # Convert a stalled game into a material-weighted outcome (mover's
            # perspective), capped so it can never match/exceed a real win.
            balance = self._material_balance(mover)
            reward += float(
                np.clip(self.truncation_material_coef * balance, -0.9, 0.9)
            )

        obs = self._get_obs()
        info = self._info()
        if self.track_info_gain:
            vis_after = visible_squares(self.board, mover)
            info["info_gain"] = float((vis_after & ~vis_before).sum())
        return obs, reward, terminated, truncated, info

    def _is_drawn(self) -> bool:
        """Non-win terminal states: only kings remain, or no move is available."""
        pieces = self.board.piece_map().values()
        if all(p.piece_type == chess.KING for p in pieces):
            return True
        if next(iter(self.board.pseudo_legal_moves), None) is None:
            return True
        return False

    def _material_balance(self, player: chess.Color) -> float:
        """Material (in pawns) from `player`'s perspective; kings excluded."""
        balance = 0.0
        for piece in self.board.piece_map().values():
            val = PIECE_VALUES.get(piece.piece_type, 0.0)
            balance += val if piece.color == player else -val
        return balance

    def _material_gain(self, move: chess.Move) -> float:
        """Material (in pawns) the mover gains from `move`: captures + promotion."""
        gain = 0.0
        victim = self.board.piece_at(move.to_square)
        if victim is not None:
            gain += PIECE_VALUES.get(victim.piece_type, 0.0)
        if self.board.is_en_passant(move):
            gain += PIECE_VALUES[chess.PAWN]
        if move.promotion:
            gain += PIECE_VALUES.get(move.promotion, 0.0) - PIECE_VALUES[chess.PAWN]
        return gain

    def current_player(self) -> chess.Color:
        """True (chess.WHITE) if it is White's turn, else False."""
        return self.board.turn

    def render(self):
        return self.board.unicode(invert_color=True, borders=True)

    def close(self):
        pass
