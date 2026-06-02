"""
Restricted negamax search over a candidate action set.

At the root, only actions in candidate_actions are explored.
Below the root, all legal moves are searched (standard alpha-beta).
Leaf evaluation uses material count from the root player's perspective.

restricted_search returns (best_action, nodes_evaluated) so callers can
track search budget for the "Search Nodes Evaluated" metric.
"""

import chess
import numpy as np

from env.board_utils import PIECE_VALUES, action_to_move, move_to_action


def _material(board: chess.Board, player: chess.Color) -> float:
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        val = PIECE_VALUES[piece.piece_type]
        score += val if piece.color == player else -val
    return score


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float,
             root_player: chess.Color, nodes: list) -> float:
    nodes[0] += 1
    if board.is_checkmate():
        return -1000.0
    if board.is_game_over():
        return 0.0
    if depth == 0:
        mat = _material(board, root_player)
        return mat if board.turn == root_player else -mat

    best = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, root_player, nodes)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def restricted_search(
    board: chess.Board,
    candidate_actions: np.ndarray,
    depth: int,
) -> tuple[int, int]:
    """
    Run negamax at depth d restricted to candidate_actions at the root.
    Operates on a copy of board so the original is never mutated.
    Returns (best_action_index, nodes_evaluated).
    """
    board = board.copy()
    player = board.turn
    legal = set(board.legal_moves)
    nodes = [0]

    best_action, best_score = None, -float('inf')

    for action in candidate_actions:
        move = action_to_move(int(action), board)
        if move not in legal:
            continue
        board.push(move)
        score = -_negamax(board, depth - 1, -float('inf'), float('inf'), player, nodes)
        board.pop()
        if score > best_score:
            best_score, best_action = score, int(action)

    if best_action is None:
        move = next(iter(board.legal_moves))
        best_action = move_to_action(move)

    return best_action, nodes[0]
