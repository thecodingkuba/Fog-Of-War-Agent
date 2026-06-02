"""
Self-play evaluation runner — pairwise and round-robin modes.

Usage:
    python evaluate.py                                          # random vs random
    python evaluate.py --white ppo --black random --model checkpoints/best_model
    python evaluate.py --white info_prior --black heuristic --model checkpoints/best_model --k 10 --depth 2
    python evaluate.py --tournament --model checkpoints/best_model --games 50
"""

import argparse
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from env.fog_chess_env import FogChessEnv

AGENT_CHOICES = ["random", "ppo", "heuristic", "policy_prior", "info_prior"]


@dataclass
class GameStats:
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    game_lengths: list[int] = field(default_factory=list)
    illegal_moves: int = 0
    total_nodes: int = 0

    @property
    def games_played(self) -> int:
        return self.white_wins + self.black_wins + self.draws

    def report(self, white_label: str = "White", black_label: str = "Black") -> str:
        n = self.games_played
        avg_len = np.mean(self.game_lengths) if self.game_lengths else 0.0
        avg_nodes = self.total_nodes / n if n else 0.0
        lines = [
            f"Games played      : {n}",
            f"{white_label:14s} wins : {self.white_wins} ({100*self.white_wins/n:.1f}%)",
            f"{black_label:14s} wins : {self.black_wins} ({100*self.black_wins/n:.1f}%)",
            f"Draws             : {self.draws} ({100*self.draws/n:.1f}%)",
            f"Avg length        : {avg_len:.1f} half-moves",
            f"Illegal moves     : {self.illegal_moves}",
            f"Avg nodes/game    : {avg_nodes:.0f}",
        ]
        return "\n".join(lines)


def _load_agent(agent_type: str, model_path: str | None, k: int, depth: int,
                info_model_path: str | None = None, info_weight: float = 0.0):
    if agent_type == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent()

    elif agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        if model_path is None:
            raise ValueError("--model required for ppo agent")
        return PPOAgent.load(model_path)

    elif agent_type == "heuristic":
        from agents.search_agents import HeuristicPriorAgent
        return HeuristicPriorAgent(k=k, depth=depth)

    elif agent_type == "policy_prior":
        from agents.search_agents import PolicyPriorAgent
        if model_path is None:
            raise ValueError("--model required for policy_prior agent")
        return PolicyPriorAgent.load(model_path, k=k, depth=depth)

    elif agent_type == "info_prior":
        from agents.search_agents import InformationPriorAgent
        # The information-aware prior is its own trained network; prefer the
        # dedicated --info-model checkpoint, falling back to --model.
        path = info_model_path or model_path
        if path is None:
            raise ValueError("--info-model (or --model) required for info_prior agent")
        return InformationPriorAgent.load(path, k=k, depth=depth,
                                          info_weight=info_weight)

    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")


def run_evaluation(
    white_agent,
    black_agent,
    n_games: int = 100,
    verbose: bool = False,
) -> GameStats:
    env = FogChessEnv()
    stats = GameStats()

    for game_idx in range(n_games):
        obs, info = env.reset()
        done = False
        half_moves = 0

        for agent in (white_agent, black_agent):
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()

        while not done:
            mask = info["action_mask"]
            agent = white_agent if env.current_player() else black_agent
            action = agent.select_action(obs, mask, env.board)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            half_moves += 1

            if info.get("illegal_move"):
                stats.illegal_moves += 1

        import chess
        last_mover = not env.current_player()
        if reward == 1.0:
            if last_mover == chess.WHITE:
                stats.white_wins += 1
            else:
                stats.black_wins += 1
        elif reward == -1.0:
            if last_mover == chess.WHITE:
                stats.black_wins += 1
            else:
                stats.white_wins += 1
        else:
            stats.draws += 1

        stats.game_lengths.append(half_moves)

        # Accumulate search nodes from both agents
        for agent in (white_agent, black_agent):
            if hasattr(agent, 'total_nodes'):
                stats.total_nodes += agent.total_nodes

        if verbose:
            print(f"Game {game_idx+1}/{n_games}: "
                  f"length={half_moves}, reward={reward:.1f}")

    env.close()
    return stats


def run_tournament(
    agents: dict[str, object],
    n_games: int = 50,
    verbose: bool = False,
) -> None:
    """
    Round-robin self-play tournament. Each pair plays n_games as White and
    n_games as Black (2*n_games total per matchup).

    Prints a win-rate table and per-agent node counts.
    """
    names = list(agents.keys())
    n = len(names)
    wins = {name: 0 for name in names}
    total_games = {name: 0 for name in names}
    total_nodes = {name: 0 for name in names}

    print(f"\nRound-robin tournament: {n} agents, {n_games} games per direction\n")

    for a_name, b_name in combinations(names, 2):
        a = agents[a_name]
        b = agents[b_name]

        # A as White, B as Black
        stats_ab = run_evaluation(a, b, n_games=n_games, verbose=verbose)
        # A as Black, B as White
        stats_ba = run_evaluation(b, a, n_games=n_games, verbose=verbose)

        a_wins = stats_ab.white_wins + stats_ba.black_wins
        b_wins = stats_ab.black_wins + stats_ba.white_wins
        played = 2 * n_games

        wins[a_name] += a_wins
        wins[b_name] += b_wins
        total_games[a_name] += played
        total_games[b_name] += played

        if hasattr(a, 'total_nodes'):
            total_nodes[a_name] += stats_ab.total_nodes + stats_ba.total_nodes
        if hasattr(b, 'total_nodes'):
            total_nodes[b_name] += stats_ab.total_nodes + stats_ba.total_nodes

        print(f"  {a_name:16s} vs {b_name:16s} | "
              f"{a_name}: {a_wins}/{played} ({100*a_wins/played:.1f}%)  "
              f"{b_name}: {b_wins}/{played} ({100*b_wins/played:.1f}%)")

    # Summary table
    print("\n--- Win Rate Summary ---")
    print(f"{'Agent':20s}  {'Win Rate':>10s}  {'Avg Nodes/Game':>16s}")
    print("-" * 52)
    for name in names:
        g = total_games[name]
        wr = 100 * wins[name] / g if g else 0.0
        nd = total_nodes[name] / g if g else 0.0
        print(f"{name:20s}  {wr:9.1f}%  {nd:>16.0f}")


def main():
    parser = argparse.ArgumentParser(description="Fog of War Chess evaluator")
    parser.add_argument("--white", default="random", choices=AGENT_CHOICES)
    parser.add_argument("--black", default="random", choices=AGENT_CHOICES)
    parser.add_argument("--model", default=None,
                        help="Path to the baseline PPO model (Dense PPO / Policy Prior)")
    parser.add_argument("--info-model", default=None,
                        help="Path to the information-trained PPO model used by "
                             "info_prior (the 'Ours' network). Falls back to --model.")
    parser.add_argument("--info-weight", type=float, default=0.0,
                        help="Optional inference-time info-gain bonus added to the "
                             "info_prior scores (ablation; the learned network is "
                             "the primary mechanism, so default 0).")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--k", type=int, default=10,
                        help="Candidate set size for search agents")
    parser.add_argument("--depth", type=int, default=2,
                        help="Minimax search depth (0 = prior only, no search)")
    parser.add_argument("--tournament", action="store_true",
                        help="Run full round-robin tournament across all agent types")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.tournament:
        tournament_agents = {}
        for name in AGENT_CHOICES:
            try:
                tournament_agents[name] = _load_agent(
                    name, args.model, args.k, args.depth,
                    info_model_path=args.info_model, info_weight=args.info_weight
                )
            except ValueError:
                if args.verbose:
                    print(f"Skipping {name} (requires --model)")
        run_tournament(tournament_agents, n_games=args.games, verbose=args.verbose)
    else:
        white = _load_agent(args.white, args.model, args.k, args.depth,
                            info_model_path=args.info_model, info_weight=args.info_weight)
        black = _load_agent(args.black, args.model, args.k, args.depth,
                            info_model_path=args.info_model, info_weight=args.info_weight)

        print(f"Running {args.games} games: {args.white} (White) vs {args.black} (Black)")
        if args.white in ("heuristic", "policy_prior", "info_prior") or \
           args.black in ("heuristic", "policy_prior", "info_prior"):
            print(f"Search: k={args.k}, depth={args.depth}")

        stats = run_evaluation(white, black, n_games=args.games, verbose=args.verbose)
        print("\n" + stats.report(args.white.capitalize(), args.black.capitalize()))


if __name__ == "__main__":
    main()
