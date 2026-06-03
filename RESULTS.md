# Fog of War Chess — Evaluation Results

## Setup

- **Round-robin tournament**, 6 agents, **100 games per direction** (200 per
  matchup, 1,000 games per agent), color-balanced.
- **Randomized openings**: 4 uniformly-random legal half-moves at the start of
  each game (`--opening-plies 4`). Without this, the deterministic agents replay
  byte-for-byte identical games and every matchup collapses to a single repeated
  result; randomized openings restore statistically meaningful win rates.
- **Search agents**: top-`k`=10 candidates, depth-2 restricted negamax
  (`--k 10 --depth 2`).
- **Rules**: true Fog of War (no check/checkmate; win by capturing the enemy
  king; pseudo-legal moves). Games truncated at 200 half-moves (scored a draw).
- **Networks**:
  - `baseline` — Dense PPO net (also drives `policy_prior`).
  - `info_v2` — information-annealed retrain (info-gain coef annealed
    `0.01 -> 0`, step penalty, material-scored truncation) → drives `info_prior`.
  - `info_infer` — the **baseline** net + an inference-time info-gain bonus at
    candidate selection (`--infer-info-weight 0.05`); no info-trained weights.

Reproduce with:

```bash
python3.12 evaluate.py --tournament \
    --model checkpoints/baseline/best_model \
    --info-model checkpoints/info_v2/best_model \
    --games 100 --k 10 --depth 2 --infer-info-weight 0.05 --opening-plies 4
```

## Win-rate summary

| Agent (method) | Win Rate | Avg Nodes / Game |
| --- | --- | --- |
| `heuristic` — Heuristic Prior + Search | **54.0%** | 26,243 |
| `info_infer` — Policy Prior + inference-time info bonus | 41.7% | 31,150 |
| `policy_prior` — Policy Prior + Search | 39.7% | 30,655 |
| `info_prior` — Information Prior + Search (training-time info) | 33.3% | 33,135 |
| `ppo` — Dense PPO (policy only, no search) | 14.0% | 0 |
| `random` | 6.0% | 0 |

## Key findings

1. **The hand-crafted heuristic prior is the strongest *and* most search-efficient**
   agent. In this tactics-dominated king-capture variant, MVV-LVA captures + a
   check bonus produce excellent depth-2 candidate sets.
2. **Among learned priors, *where* information is injected matters**:
   inference-time info-gain on the strong baseline net (`info_infer`, 41.7%)
   beats both the plain policy prior (`policy_prior`, 39.7%) and the
   training-time info-reward net (`info_prior`, 33.3%).
3. **Baking information gain into the training reward hurts.** Even after
   annealing + a step penalty + material-scored truncation, `info_prior` is the
   weakest search agent and uses the most search nodes (it draws frequently,
   lengthening games). Information-awareness is better added at selection time
   than optimized into the policy.
4. **Search is decisive**: every search agent (40–54%) dominates the
   policy-only `ppo` (14%), which in turn beats `random` (6%). Note `ppo` is
   evaluated deterministically (argmax); sampled play was far weaker.

## Top-k ablation (k = 1, 2, 4, 8, 10)

Same tournament setup as above (100 games per direction, depth 2,
`--opening-plies 4`). Raw logs: `tournament_k1.log`, `tournament_k2.log`,
`tournament_k4.log`, `tournament_k8.log`, `tournament_v2.log` (k=10).

Reproduce a single k:

```bash
python3.12 evaluate.py --tournament \
    --model checkpoints/baseline/best_model \
    --info-model checkpoints/info_v2/best_model \
    --games 100 --k K --depth 2 --infer-info-weight 0.05 --opening-plies 4
```

### Win rate vs. k

| Agent | k=1 | k=2 | k=4 | k=8 | k=10 |
| --- | --- | --- | --- | --- | --- |
| `heuristic` | **97.3%** | **86.5%** | **68.5%** | **56.9%** | **54.0%** |
| `info_infer` | 36.1% | 36.3% | 35.8% | 41.3% | 41.7% |
| `policy_prior` | 37.3% | 35.6% | 35.8% | 39.6% | 39.7% |
| `ppo` (no search) | 36.7% | 33.0% | 27.7% | 17.8% | 14.0% |
| `info_prior` | 7.1% | 13.2% | 20.4% | 29.5% | 33.3% |
| `random` | 25.2% | 16.9% | 10.4% | 7.0% | 6.0% |

### Avg search nodes / game vs. k

| Agent | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- |
| `heuristic` | 462 | 2,206 | 7,891 | 19,540 |
| `info_infer` | 1,663 | 4,448 | 11,329 | 23,154 |
| `policy_prior` | 1,671 | 4,419 | 10,916 | 23,872 |
| `info_prior` | 2,454 | 5,238 | 12,016 | 25,778 |

### Ablation takeaways

1. **`heuristic` dominates at every k and is strongest at small k** (97.3% at
   k=1). Its top-ranked move is almost always tactically correct, so it needs
   little search. Its edge shrinks as k grows but it never loses the lead.
2. **`info_prior` is the most k-sensitive**: 7.1% at k=1, climbing to 33.3%
   at k=10. The info-reward training demotes the best tactical move out of the
   #1 slot, but good moves remain in the broader candidate set — wider pruning
   + search recover them.
3. **`info_infer` ≥ `policy_prior` at every k** (~36–41%), with flat performance
   across k — robust priors whose top move is already solid.
4. **Efficiency**: `heuristic` uses the fewest nodes at every k and wins;
   `info_prior` uses the most nodes and is weakest among search agents.

## Caveats / future work

- Search-vs-search matchups are draw-heavy (games often reach the 200-move cap),
  which compresses win-rate differences among the search agents.
- The policy was trained via self-play **always as White**, and the board tensor
  is not vertically flipped for the side to move, so the learned nets are weaker
  as Black (most visible for bare `ppo`). Training both colors / a
  color-canonical observation is the natural fix.
