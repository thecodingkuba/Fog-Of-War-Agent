"""
Move encoding, action masks, visibility, and heuristic pruning utilities.

Action encoding
---------------
A move is encoded as a single integer in [0, 4096):

    action = from_square * 64 + to_square

This "from-to" scheme collapses all promotions to a single action per
(from, to) pair; we always promote to a queen, which is the standard
simplification used by most chess-RL setups. The action space therefore has
size 64 * 64 = 4096.
"""

from __future__ import annotations

import chess
import numpy as np

ACTION_SPACE_SIZE = 64 * 64  # 4096

# Material values (king = 0 so it is ignored by material-count evaluations).
PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Channel order for piece planes in the observation tensor.
PIECE_TO_PLANE: dict[int, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def move_to_action(move: chess.Move) -> int:
    """Encode a chess.Move into an integer action index in [0, 4096)."""
    return move.from_square * 64 + move.to_square


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """
    Decode an action index into a concrete legal chess.Move on `board`.

    Because promotions are collapsed to one action, we look up the matching
    legal move and prefer queen promotion when several promotions share the
    same (from, to) pair. Falls back to a bare Move if nothing matches.
    """
    from_sq = action // 64
    to_sq = action % 64
    candidates = [
        m for m in board.legal_moves
        if m.from_square == from_sq and m.to_square == to_sq
    ]
    if not candidates:
        return chess.Move(from_sq, to_sq)
    for m in candidates:
        if m.promotion == chess.QUEEN:
            return m
    return candidates[0]


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (4096,); True where a legal move exists."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = True
    return mask


def visible_squares(board: chess.Board, color: chess.Color) -> np.ndarray:
    """
    64-element boolean visibility mask for `color` under fog of war.

    A square is visible if it holds one of `color`'s own pieces or is
    attacked by one of them. Attack geometry (board.attacks) is independent
    of whose turn it is, so this works for either side at any time.
    """
    mask = np.zeros(64, dtype=bool)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            mask[sq] = True
            for attacked in board.attacks(sq):
                mask[attacked] = True
    return mask


def heuristic_move_scores(board: chess.Board) -> dict[chess.Move, float]:
    """
    Cheap domain heuristic used for action-space pruning.

    Score = captured material (MVV) + a small bonus for giving check. This is
    the "simple heuristic" described in the proposal used to rank moves before
    keeping only the top-k candidates.
    """
    scores: dict[chess.Move, float] = {}
    for move in board.legal_moves:
        s = 0.0
        victim = board.piece_at(move.to_square)
        if victim is not None:
            s += PIECE_VALUES.get(victim.piece_type, 0.0)
        if board.is_en_passant(move):
            s += PIECE_VALUES[chess.PAWN]
        board.push(move)
        if board.is_check():
            s += 0.5
        board.pop()
        scores[move] = s
    return scores


def top_k_action_mask(board: chess.Board, k: int) -> np.ndarray:
    """
    Pruned action mask: keep only the top-k legal moves by heuristic score.

    If there are <= k legal moves, this is identical to legal_action_mask.
    Ties are broken deterministically by the move ordering python-chess yields.
    """
    legal = list(board.legal_moves)
    if len(legal) <= k:
        return legal_action_mask(board)

    scores = heuristic_move_scores(board)
    best = sorted(legal, key=lambda m: scores[m], reverse=True)[:k]
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in best:
        mask[move.from_square * 64 + move.to_square] = True
    return mask
