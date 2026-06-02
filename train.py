"""
Recurrent PPO self-play training for Fog of War Chess.

Usage:
    python train.py
    python train.py --timesteps 2000000 --eval-games 50 --checkpoint-freq 50000
    python train.py --resume checkpoints/best_model

Architecture:
  - White is the learning agent (RecurrentPPO + MaskedRecurrentPolicy)
  - Black is a frozen opponent snapshot, updated every `opponent_update_freq` steps
  - Masking is applied inside MaskedRecurrentPolicy, not via ActionMasker

TensorBoard logs → ./logs/
Checkpoints     → ./checkpoints/
"""

import argparse
import os

import numpy as np
from gymnasium import Env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.fog_chess_env import FogChessEnv
from env.board_utils import legal_action_mask
from models.policy_net import FogChessCNN, MaskedRecurrentPolicy

# ---------------------------------------------------------------------------
# Self-play wrapper
# ---------------------------------------------------------------------------

class SelfPlayEnv(Env):
    """
    Wraps FogChessEnv so the learning agent always plays as White.
    Black is a frozen opponent; lstm_states for black are tracked here
    so the opponent also benefits from recurrent memory.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None):
        super().__init__()
        self._env = FogChessEnv()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._opponent = opponent_model
        self._opp_lstm_states = None
        self._opp_episode_start = np.array([True])

    def set_opponent(self, model) -> None:
        self._opponent = model

    def _opponent_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self._opponent is None:
            return int(np.random.choice(np.where(mask)[0]))

        # Feed mask into policy before predict
        if hasattr(self._opponent.policy, "set_action_mask"):
            import torch
            self._opponent.policy.set_action_mask(
                torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            )

        action, self._opp_lstm_states = self._opponent.predict(
            obs[np.newaxis],
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
        # Inject the action mask so RecurrentPPO can read it each step
        self._inject_mask(info)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            self._inject_mask(info)
            return obs, reward, terminated, truncated, info

        black_mask = info["action_mask"]
        black_action = self._opponent_action(obs, black_mask)
        obs, black_reward, terminated, truncated, info = self._env.step(black_action)

        white_reward = -black_reward
        self._inject_mask(info)
        return obs, white_reward, terminated, truncated, info

    def _inject_mask(self, info: dict) -> None:
        """Store the current legal mask so the training loop can retrieve it."""
        info["action_mask"] = legal_action_mask(self._env.board)

    def get_action_mask(self) -> np.ndarray:
        return legal_action_mask(self._env.board)

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Callback: periodically snapshot the current policy as the new opponent
# ---------------------------------------------------------------------------

class OpponentUpdateCallback(BaseCallback):
    def __init__(self, update_freq: int, env: SelfPlayEnv, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self._self_play_env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._self_play_env.set_opponent(self.model)
            if self.verbose:
                print(f"[Step {self.n_calls}] Opponent updated.")
        return True


# ---------------------------------------------------------------------------
# Mask injection callback — sets the legal mask on the policy before each step
# ---------------------------------------------------------------------------

class MaskInjectionCallback(BaseCallback):
    """Reads the action_mask from the env info and passes it to the policy."""

    def _on_step(self) -> bool:
        import torch
        masks = self.locals.get("infos", [{}])
        if masks and "action_mask" in masks[0]:
            mask = np.stack([info["action_mask"] for info in masks])
            self.model.policy.set_action_mask(
                torch.tensor(mask, dtype=torch.bool)
            )
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_env(opponent_model=None):
    def _init():
        env = SelfPlayEnv(opponent_model)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--opponent-update-freq", type=int, default=10_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_env = DummyVecEnv([make_env(opponent_model=None)])

    policy_kwargs = dict(
        features_extractor_class=FogChessCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
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
            verbose=1,
            tensorboard_log="logs/",
        )

    eval_env = DummyVecEnv([make_env(opponent_model=None)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/",
        log_path="logs/",
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=args.eval_games,
        deterministic=False,
        render=False,
    )

    # inner_env path: DummyVecEnv -> Monitor -> SelfPlayEnv
    inner_env = train_env.envs[0].env
    opponent_callback = OpponentUpdateCallback(
        update_freq=args.opponent_update_freq,
        env=inner_env,
        verbose=1,
    )

    mask_callback = MaskInjectionCallback()

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, opponent_callback, mask_callback],
        reset_num_timesteps=not bool(args.resume),
    )

    model.save("checkpoints/final_model")
    print("Training complete. Model saved to checkpoints/final_model")


if __name__ == "__main__":
    main()
