# Fog-Of-War-Agent

Reinforcement learning for **Fog of War Chess** with action-space pruning.

This is the codebase for the CS224R project _"Reinforcement Learning for Fog of
War Chess with Action Space Pruning."_ We train a PPO agent to play a partially
observable variant of chess and study whether pruning the action space to a
small set of heuristically "reasonable" moves makes RL more efficient.

## What's implemented

- **`env/`** — a Gymnasium Fog of War Chess environment built on `python-chess`.
  True FoW rules (no check/checkmate; win by capturing the enemy king;
  pseudo-legal moves), and each player only observes the squares its own pieces
  can see. The observation is a `Dict`:
  - `board`: `(14, 8, 8)` tensor — own pieces (6 planes), visible enemy pieces
    (6 planes), known-empty squares (1 plane), and fog/unknown squares (1 plane),
    all from the side-to-move's perspective.
  - `action_mask`: `(4096,)` vector of allowed actions. Either the full legal
    move set (baseline) or the top-k heuristic candidates (pruning).

  Moves are encoded as `from_square * 64 + to_square` (promotions default to
  queen), giving a `Discrete(4096)` action space.

- **`models/policy_net.py`** — a CNN feature extractor plus a recurrent
  (LSTM) actor-critic policy. The legal/pruned action mask is carried **inside
  the observation**, so masking stays consistent between rollout collection and
  gradient updates (a subtle requirement when combining masking with
  `RecurrentPPO`).

- **`agents/`** — opponents and evaluation agents: `RandomAgent`,
  `PPOAgent`, and three prior-based search agents (`HeuristicPriorAgent`,
  `PolicyPriorAgent`, `InformationPriorAgent`) that rank moves, keep the top-k,
  and run a restricted minimax search. `PolicyPriorAgent` uses the baseline
  PPO policy as the prior; `InformationPriorAgent` uses a *separately
  trained* information-seeking policy (see `--info-reward-coef`), optionally
  plus an inference-time information-gain bonus for ablations.

- **`search/`** — restricted negamax (`minimax.py`) and fog-of-war information
  gain scoring (`information_gain.py`).

- **`train.py`** — RecurrentPPO self-play training, supporting the two
  experimental conditions from the proposal.

- **`evaluate.py`** — head-to-head and round-robin tournament evaluation.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Training

Baseline PPO with the full (legal) action space:

```bash
python train.py --tag baseline
```

Information-aware prior (PPO self-play with an added information-gain reward,
so the learned prior seeks uncertainty-reducing moves). The reported model
anneals the info reward toward 0 and adds a step penalty plus material-based
truncation scoring so the prior seeks information early but ultimately learns
to win:

```bash
python train.py --info-reward-coef 0.01 --info-reward-coef-final 0.0 \
    --step-penalty 0.001 --truncation-material-coef 0.05 --tag info_v2
```

PPO with action-space pruning during training (keep top-k=10 heuristic moves
each step — a separate, earlier ablation):

```bash
python train.py --prune-k 10 --tag pruned
```

### Which checkpoints feed which method

The four methods from the poster differ only in the **prior** that drives the
top-k search, so they need **two** trained networks (plus the model-free
heuristic):

| Method | Agent | Checkpoint |
| --- | --- | --- |
| Dense PPO | `ppo` | `checkpoints/baseline` |
| Policy Prior + Search | `policy_prior` | `checkpoints/baseline` |
| Heuristic Prior + Search | `heuristic` | none |
| Information Prior + Search | `info_prior` | `checkpoints/info_v2` |

```bash
python train.py --tag baseline   # Dense PPO + Policy Prior
python train.py --info-reward-coef 0.01 --info-reward-coef-final 0.0 \
    --step-penalty 0.001 --truncation-material-coef 0.05 --tag info_v2  # Information Prior
```

Useful flags: `--timesteps`, `--eval-games`, `--checkpoint-freq`,
`--opponent-update-freq`, `--shaping-coef`, `--lr`, `--n-steps`,
`--batch-size`, `--seed`, `--resume <path>`.

**Reward shaping.** Checkmate is rare under fog of war, so a pure win/loss
reward is too sparse to learn from. By default training adds dense material
shaping: `reward += shaping_coef * (material captured/promoted)`, where the
self-play wrapper also subtracts the opponent's gains (so losing material is
penalized). Set `--shaping-coef 0` for the pure sparse reward. Shaping only
affects training/eval reward; the standalone `evaluate.py` reports clean
win/draw/loss outcomes.

**Frozen self-play opponent.** The opponent is a *deep-copied snapshot* of the
policy, refreshed every `--opponent-update-freq` steps. (Playing the live model
would be exact self-play symmetry, pinning mean reward to ~0 regardless of
skill.) The learner climbs by beating its older, frozen self.

Checkpoints are written to `checkpoints/<tag>/` and TensorBoard logs to
`logs/<tag>/`:

```bash
tensorboard --logdir logs/
```

## Evaluation

```bash
# Random vs random sanity check
python evaluate.py --games 100

# Trained PPO vs a random opponent
python evaluate.py --white ppo --black random --model checkpoints/baseline/best_model --games 100

# Information-prior search agent (uses its own info-trained network)
python evaluate.py --white info_prior --black policy_prior \
    --info-model checkpoints/info_v2/best_model --k 10 --depth 2 --games 100

# Full round-robin tournament: baseline net for ppo/policy_prior, info net for info_prior
python evaluate.py --tournament \
    --model checkpoints/baseline/best_model \
    --info-model checkpoints/info_v2/best_model \
    --games 50 --k 10 --depth 2
```

Reported metrics: win/draw rates, average game length, illegal-move count
(should be 0), and average search nodes per game.

## Notes

- The environment uses true **Fog of War rules**: no check or checkmate — a
  player may move into check and the game is **won by capturing the enemy
  king**. Moves are pseudo-legal (king captures and moving into check are
  allowed). This keeps games decisive (low draw rate), so Win Rate is a
  meaningful metric. Games are truncated at 200 half-moves (scored as a draw),
  and positions with only the two kings left are also draws.
- Self-play uses a frozen opponent snapshot refreshed every
  `--opponent-update-freq` steps; before the first refresh the opponent moves
  randomly.
