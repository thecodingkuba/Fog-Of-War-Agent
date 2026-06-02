"""
FoW information gain: how many new squares does a move reveal?

Visibility is approximated via attack geometry (board.attacks), which works
regardless of whose turn it is — unlike legal_moves which requires board.turn
to match the player.
"""

import chess
import numpy as np

from env.board_utils import move_to_action


def _visibility(board: chess.Board, player: chess.Color) -> np.ndarray:
    """64-element bool mask of squares visible to player."""
    mask = np.zeros(64, dtype=bool)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == player:
            mask[sq] = True
            for attacked in board.attacks(sq):
                mask[attacked] = True
    return mask


def information_gain(board: chess.Board, move: chess.Move) -> int:
    """Number of new squares revealed to the moving player by making this move."""
    player = board.turn
    before = _visibility(board, player)

    board_copy = board.copy()
    board_copy.push(move)
    after = _visibility(board_copy, player)

    return int((after & ~before).sum())


def information_gain_scores(board: chess.Board) -> np.ndarray:
    """Return a (4096,) float32 array of information gain for all legal moves."""
    scores = np.zeros(4096, dtype=np.float32)
    for move in board.pseudo_legal_moves:
        scores[move_to_action(move)] = information_gain(board, move)
    return scores
