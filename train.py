"""
Recurrent PPO self-play training for Fog of War Chess.

Usage:
    python train.py                                  # baseline (full action space)
    python train.py --prune-k 10 --tag pruned        # PPO with top-k action pruning
    python train.py --timesteps 2000000 --eval-games 50 --checkpoint-freq 50000
    python train.py --resume checkpoints/baseline/best_model

Architecture:
  - White is the learning agent (RecurrentPPO + MaskedRecurrentPolicy).
  - Black is a frozen opponent snapshot, refreshed every `opponent_update_freq`
    steps (self-play). Until the first refresh the opponent plays randomly.
  - Legal-move masking (and optional top-k pruning) is carried inside the
    observation's "action_mask" key, so it stays consistent between rollout
    collection and gradient updates.

Two experimental conditions from the proposal:
  - Baseline PPO with the full (legal) action space   -> --prune-k unset
  - PPO with action-space pruning (top-k candidates)   -> --prune-k K

TensorBoard logs -> ./logs/<tag>/
Checkpoints      -> ./checkpoints/<tag>/
"""

import argparse
import copy
import os

import numpy as np
from gymnasium import Env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.fog_chess_env import FogChessEnv
from models.policy_net import FogChessCNN, MaskedRecurrentPolicy

# ---------------------------------------------------------------------------
# Self-play wrapper
# ---------------------------------------------------------------------------


class SelfPlayEnv(Env):
    """
    Wraps FogChessEnv so the learning agent always plays as White.

    Black is a frozen opponent snapshot; its LSTM state is tracked here so the
    opponent also benefits from recurrent memory across its own turns.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None, prune_k=None, shaping_coef=0.0,
                 info_reward_coef=0.0):
        super().__init__()
        # Information-gain shaping is NOT zero-sum, so it is applied here to the
        # learner (White) only rather than inside the negated env reward.
        self.info_reward_coef = info_reward_coef
        self._env = FogChessEnv(
            prune_k=prune_k,
            shaping_coef=shaping_coef,
            track_info_gain=info_reward_coef > 0.0,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._opponent = opponent_model
        self._opp_lstm_states = None
        self._opp_episode_start = np.array([True])

    def set_opponent(self, model) -> None:
        self._opponent = model

    @staticmethod
    def _batch_obs(obs: dict) -> dict:
        return {k: v[np.newaxis, ...] for k, v in obs.items()}

    def _opponent_action(self, obs: dict) -> int:
        mask = obs["action_mask"] > 0.5
        if self._opponent is None:
            return int(np.random.choice(np.where(mask)[0]))

        # The masked policy reads the legal mask straight from the observation.
        action, self._opp_lstm_states = self._opponent.predict(
            self._batch_obs(obs),
            state=self._opp_lstm_states,
            episode_start=self._opp_episode_start,
            deterministic=False,
        )
        self._opp_episode_start = np.array([False])
        return int(action)

    def reset(self, *, seed=None, options=None):
        self._opp_lstm_states = None
        self._opp_episode_start = np.array([True])
        obs, info = self._env.reset()
        return obs, info

    def step(self, action: int):
        # White (the learner) moves.
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Reward White for the information its own move revealed (not zero-sum).
        white_info = self.info_reward_coef * info.get("info_gain", 0.0)
        if terminated or truncated:
            return obs, reward + white_info, terminated, truncated, info

        # Black (the frozen opponent) responds.
        black_action = self._opponent_action(obs)
        obs, black_reward, terminated, truncated, info = self._env.step(black_action)

        # Material/check shaping is zero-sum, so negate Black's; the info-gain
        # bonus only credits White's own move.
        white_reward = reward - black_reward + white_info
        return obs, white_reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Callback: periodically snapshot the current policy as the new opponent
# ---------------------------------------------------------------------------


class OpponentUpdateCallback(BaseCallback):
    """Periodically freeze a *copy* of the current policy as the opponent.

    Crucially we deep-copy the policy weights rather than handing over a live
    reference to the model. Playing a frozen snapshot (that lags the learner)
    is what gives self-play a real gradient to climb; playing the live model
    is exact self-play symmetry, which pins the mean reward to ~0 regardless
    of how much the agent improves.
    """

    def __init__(self, update_freq: int, envs, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self._envs = envs if isinstance(envs, (list, tuple)) else [envs]

    def _snapshot(self):
        frozen = copy.deepcopy(self.model.policy)
        frozen.set_training_mode(False)
        return frozen

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            frozen = self._snapshot()
            for env in self._envs:
                env.set_opponent(frozen)
            if self.verbose:
                print(f"[Step {self.n_calls}] Opponent snapshot frozen.")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_env(opponent_model=None, prune_k=None, shaping_coef=0.0,
             info_reward_coef=0.0):
    def _init():
        env = SelfPlayEnv(
            opponent_model,
            prune_k=prune_k,
            shaping_coef=shaping_coef,
            info_reward_coef=info_reward_coef,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--opponent-update-freq", type=int, default=10_000)
    parser.add_argument("--prune-k", type=int, default=None,
                        help="If set, prune the action space to the top-k "
                             "heuristic moves each step (PPO-with-pruning).")
    parser.add_argument("--tag", default=None,
                        help="Run name for logs/checkpoints "
                             "(defaults to 'baseline' or 'pruned_k<K>').")
    parser.add_argument("--shaping-coef", type=float, default=0.01,
                        help="Dense material reward shaping coefficient "
                             "(0 disables; reward += coef * material gained).")
    parser.add_argument("--info-reward-coef", type=float, default=0.0,
                        help="Information-gain reward coefficient. >0 trains an "
                             "information-seeking prior (the 'Ours' network): "
                             "reward += coef * (newly revealed squares). "
                             "Use e.g. --info-reward-coef 0.02 --tag info.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.tag:
        tag = args.tag
    elif args.info_reward_coef > 0:
        tag = "info"
    elif args.prune_k:
        tag = f"pruned_k{args.prune_k}"
    else:
        tag = "baseline"
    ckpt_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = DummyVecEnv([
        make_env(opponent_model=None, prune_k=args.prune_k,
                 shaping_coef=args.shaping_coef,
                 info_reward_coef=args.info_reward_coef)
    ])

    policy_kwargs = dict(
        features_extractor_class=FogChessCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = RecurrentPPO.load(args.resume, env=train_env)
    else:
        model = RecurrentPPO(
            MaskedRecurrentPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            seed=args.seed,
            verbose=1,
            tensorboard_log=log_dir,
        )

    eval_env = DummyVecEnv([
        make_env(opponent_model=None, prune_k=args.prune_k,
                 shaping_coef=args.shaping_coef,
                 info_reward_coef=args.info_reward_coef)
    ])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=log_dir,
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=args.eval_games,
        deterministic=False,
        render=False,
    )

    # inner_env path: DummyVecEnv -> Monitor -> SelfPlayEnv
    inner_env = train_env.envs[0].env
    opponent_callback = OpponentUpdateCallback(
        update_freq=args.opponent_update_freq,
        envs=[inner_env],
        verbose=1,
    )

    print(f"[{tag}] Training for {args.timesteps:,} timesteps "
          f"(prune_k={args.prune_k}, shaping_coef={args.shaping_coef}, "
          f"info_reward_coef={args.info_reward_coef})...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, opponent_callback],
        reset_num_timesteps=not bool(args.resume),
    )

    final_path = os.path.join(ckpt_dir, "final_model")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
