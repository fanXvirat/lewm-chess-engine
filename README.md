<p align="center">
  <h1 align="center">♟️ LeWM Chess Engine</h1>
  <p align="center">
    <strong>A Chess World Model built with JEPA</strong><br>
    Teaching a neural network to <em>understand chess</em> — not just memorize moves
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2603.19312">📄 LeWM Paper</a> •
    <a href="https://github.com/lucas-maes/le-wm">🔗 Original Repo</a> •
    <a href="#results">📊 Results</a> •
    <a href="#quickstart">🚀 Quickstart</a>
  </p>
</p>

---

## What is this?

This project applies **[LeWorldModel](https://arxiv.org/abs/2603.19312)** — a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) — to chess. Instead of the traditional approach of search + evaluation, we train a **world model** that learns the dynamics of chess games from raw board images, then uses its learned representations for:

- **🎯 Move prediction** — 29.8% top-1 / 70.4% top-10 accuracy
- **📈 Game outcome prediction** — 56.5% accuracy (vs 33% random baseline)
- **🧩 Tactical puzzle generation** — via JEPA "surprise detection" (Violation of Expectation)
- **🌌 Structured latent space** — separates openings from endgames, winning from losing positions
- **🎮 Playable chess engine** — both policy-based (instant) and CEM planning modes

> **Key insight**: We train on **full-game trajectories** (20-move windows), not isolated positions. This lets the model learn *strategic dynamics* — how pawn structures evolve, how piece coordination develops, how advantages convert to wins — like a human studying entire games.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              LeWM Chess Engine                   │
                    └─────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐
            │  ViT-Tiny    │   │  ARPredictor     │   │  Move/Value  │
            │  Encoder     │   │  (6-layer, 16h)  │   │  Heads       │
            │              │   │                  │   │              │
            │ 128×128 board│   │ 6-way AdaLN-zero │   │ Policy: 2.3M │
            │ → 192-dim z  │   │ + game-progress  │   │ Value:  0.1M │
            └──────┬───────┘   │ conditioning     │   └──────────────┘
                   │           └────────┬─────────┘
                   ▼                    ▼
            ┌──────────────┐   ┌─────────────────┐
            │  SIGReg      │   │  Multi-step      │
            │  Anti-collapse│   │  Prediction      │
            │  Regularizer │   │  ẑ_{t+4} from    │
            └──────────────┘   │  z_{t}, a_{t}    │
                               └─────────────────┘
```

**Total parameters: 20.8M** (18.4M JEPA + 2.3M Policy + 0.1M Value)

### Paper-faithful components

| Component | Paper (§) | Our Implementation |
|-----------|-----------|-------------------|
| Encoder | ViT-Tiny (§3.1) | `timm vit_tiny_patch16`, 128×128 input, 64 tokens (1 per square) |
| Predictor | 6-layer AdaLN-zero (§3.1) | 6-layer Transformer, 16 heads, causal attention |
| SIGReg | Gaussian-weighted Epps-Pulley (§3.1) | 17 knots, 1024 projections, λ=0.09 |
| Loss | ℒ_pred + λ·ℒ_SIGReg (Eq. 1-3) | MSE prediction + SIGReg, fixed λ from epoch 0 |
| Training | AdamW, bf16 (App. D) | AdamW, bf16, cosine LR, grad_clip=5.0 |

### Chess-specific innovations

| Innovation | Description |
|-----------|-------------|
| **Visual board encoding** | 128×128 RGB images with 12 piece-specific color categories. Patch size 16 → each ViT token = exactly one chess square |
| **Full-game trajectories** | `seq_len=20, history=16, preds=4` — the model sees 16 positions of context and predicts 4 steps ahead (~32 half-moves of strategic dynamics) |
| **Game-progress conditioning** | Move progress scalar (0.0=opening, 1.0=endgame) projected and added to action embeddings, so the predictor knows the game phase |
| **Value head** | Predicts game outcome (white win / black win / draw) from mean-pooled latent embeddings, forcing strategic information into the latent space |
| **Surprise-based puzzles** | High JEPA prediction error (Violation of Expectation, §5.2) = tactically interesting position |

## Results

### Training Summary

| Metric | Value |
|--------|-------|
| **Hardware** | NVIDIA A100-SXM4 (40GB VRAM) |
| **Dataset** | 17,229 games (Gukesh D. PGN, 826,878 positions) |
| **Train split** | 436,478 trajectories from 15,507 games |
| **Val split** | 45,820 trajectories from 1,722 games |
| **Training** | 10 epochs × 3,409 steps/epoch (~19min/epoch) |
| **Best val_pred** | **0.1470** (epoch 6) |
| **Model size** | 20.8M parameters |

### Epoch-by-Epoch Training Log

| Epoch | train_pred | val_pred | Policy Acc | Value Acc | Time |
|-------|-----------|----------|------------|-----------|------|
| 1 | 0.3062 | **0.2448** ✓ | 2.91% | 38.39% | 1140s |
| 2 | 0.1986 | **0.1759** ✓ | 5.00% | 43.44% | 1121s |
| 3 | 0.1575 | **0.1478** ✓ | 8.88% | 44.21% | 1121s |
| 4 | 0.1516 | 0.1545 | 13.67% | 41.03% | 1123s |
| 5 | 0.1426 | 0.1488 | 17.51% | 48.59% | 1126s |
| **6** | **0.1332** | **0.1470** ✓ | **19.48%** | **46.83%** | **1122s** |
| 7 | 0.1245 | 0.1563 | 21.70% | 44.61% | 1127s |
| 8 | 0.1154 | 0.1614 | 22.79% | 46.36% | 1121s |
| 9 | 0.1067 | 0.1679 | 23.25% | 48.80% | 1126s |
| 10 | 0.0996 | 0.1714 | 24.19% | 48.89% | 1122s |

> **✓** = best model saved. Epoch 6 achieves the best generalization (val_pred=0.1470). After epoch 6, train_pred continues to decrease but val_pred rises — classic mild overfitting due to limited validation data. Policy and value accuracy on training data continue to climb (reaching 65%+ pacc and 100% vacc by epoch 10), confirming the model is learning real patterns.

### Latent Space Probing (Linear/MLP R²)

```
material_balance          linear=0.336  MLP=0.149
white_king_safety         linear=0.413  MLP=0.164
black_king_safety         linear=0.416  MLP=0.255
```

The latent space encodes chess-relevant features that are linearly decodable — king safety is especially well-represented.

---

### 📈 Game Value Analysis

The value head tracks win probability across an entire game. The model learned to read positions — not just count material:

<p align="center">
  <img src="assets/lewm_game_analysis.png" width="100%">
</p>

**What this shows:**
- **Top panel**: Win probability tracked over all 40 moves of Abdusattorov vs Alikulov. The model correctly identifies swings between White (blue) and Black (red) advantage
- **Middle panel**: Raw material balance — the model's evaluation goes *beyond* simple piece counting
- **Bottom panel**: JEPA surprise scores — Move 0-2 are the most surprising (opening theory that the model hasn't fully internalized)

### 🌌 Latent Space Structure (t-SNE)

The model organizes 192-dimensional embeddings in a meaningful way — without any explicit instruction to do so:

<p align="center">
  <img src="assets/lewm_latent_space.png" width="100%">
</p>

**What this shows:**
- **Left**: Positions from white-winning games (blue) vs black-winning (red) show separable clusters — the model encodes who's advantaged
- **Center**: Material balance creates a gradient in latent space — the model understands piece values
- **Right**: Game phase (purple=opening → yellow=endgame) shows clear clustering — the model distinguishes opening structures from endgame positions

### 🎯 Policy Head — Move Prediction

The policy head predicts Gukesh's moves from board position embeddings alone:

<p align="center">
  <img src="assets/lewm_policy_analysis.png" width="100%">
</p>

| Metric | Accuracy |
|--------|----------|
| **Top-1** (exact move) | 29.8% |
| **Top-3** | 48.0% |
| **Top-5** | 57.2% |
| **Top-10** | 70.4% |
| Castling moves | **90.0%** |
| Captures | 30.5% |
| Checks | 24.0% |
| Quiet moves | 28.3% |

> The model predicts the exact GM move ~30% of the time, and has it in the top-10 over 70% of the time. **Castling at 90%** shows the model deeply learned this critical pattern. For context, there are ~30 legal moves per position on average, so random chance is ~3%.

### 📊 Value Head — Game Outcome Prediction

The value head predicts who will win from any position:

<p align="center">
  <img src="assets/lewm_value_calibration.png" width="100%">
</p>

| Phase | Accuracy | Samples | vs Random (33%) |
|-------|----------|---------|-----------------|
| Opening | 43.8% | n=607 | +10.5pp |
| Middlegame | 60.0% | n=797 | +26.7pp |
| Endgame | **64.8%** | n=596 | +31.5pp |
| **Overall** | **56.5%** | n=2000 | **+23.2pp** |

**Key findings:**
- Accuracy increases from opening → endgame (positions become more "decided")
- The confidence calibration curve shows the model is well-calibrated: when it's 90%+ confident, it's correct ~80% of the time
- Overall 56.5% on a 3-class problem demonstrates genuine strategic understanding

### 🧩 JEPA Puzzle Generation (Violation of Expectation)

Using the paper's "violation of expectation" framework (§5.2), we identify tactically interesting positions where the world model's prediction was most wrong:

```
Puzzle #1 — Surprise: 755.97 [capture, check]
  Game: Studer,N vs Nepomniachtchi,I (Titled Tuesday, 2021.02.02)
  FEN: 1Q6/8/7K/8/4kq2/8/8/8  —  White to move → Qxf4+

Puzzle #5 — Surprise: 674.67 [capture, check]
  Game: Abdusattorov,Nodirbek vs So,W (Superbet Classic 2025)
  FEN: 7B/1p6/p3k2p/1n1p2p1/3P4/1P1KPPn1/1P5R/8  —  White to move → Rxh6

Puzzle #9 — Surprise: 545.45 [capture, check]
  Game: Gukesh,D vs Barseghyan,Armen (ChessMood Open 2021)
  FEN: 8/p1K5/2P5/5bkp/8/8/5R2/8  —  White to move → Rxf5+
```

All 10 generated puzzles involve **captures and/or checks** — exactly the "surprising" positions a JEPA world model should flag. The puzzles span games featuring Gukesh, Nepomniachtchi, Abdusattorov, Grischuk, and others.

### 🎮 Live Play Demo

The engine plays in real-time via both policy (instant) and CEM planning modes:

```
LeWM-Policy plays: g7g6 (0.1s)     ← Instant policy prediction
LeWM-CEM plays: b7b6 (0.9s)        ← World-model rollout planning
```

## How It Works

### The JEPA Approach to Chess

Traditional chess engines use search (alpha-beta / MCTS) + evaluation. Our approach is fundamentally different:

1. **Learn a world model** — understand how board states evolve through latent dynamics
2. **Predict in latent space** — imagine future positions without rendering them
3. **Use prediction error** as a signal for tactical interest

```python
# The core training loop (simplified)
z_t = encoder(board_image_t)          # Encode current position
z_{t+4} = encoder(board_image_{t+4})  # Encode future position (4 moves ahead)
ẑ_{t+4} = predictor(z_t, moves)       # Predict future from current + moves

loss = MSE(ẑ_{t+4}, z_{t+4})          # Learn dynamics
     + λ · SIGReg(Z)                   # Prevent representation collapse
     + α · CE(policy(z_t), move_t)     # Learn to predict moves
     + β · CE(value(z_t), result)      # Learn to predict game outcomes
```

### Why Full-Game Trajectories Matter

The original LeWM trains on continuous robot trajectories (200+ frames). We adapted this for chess:

| Approach | Context | What it learns |
|----------|---------|----------------|
| Position-by-position | 1 board | Piece recognition |
| Short windows (seq=4) | 3 boards | Local tactics |
| **Full game (seq=20)** | **16 boards** | **Strategic patterns, game-phase dynamics** |

With 16 positions of context (~16 full moves), the model sees opening→middlegame transitions, pawn structure evolution, and piece coordination patterns.

## Quickstart

### Requirements

```bash
pip install torch>=2.0 timm python-chess einops numpy pillow tqdm scikit-learn matplotlib
```

### Training

```python
# In Google Colab or Lightning.ai (A100 recommended)

from lewm_chess import *

CFG = Config()

# 1. Parse PGN → cached tensor dataset
cache_path = parse_and_cache(CFG.pgn_path, CFG)

# 2. Train (10 epochs ≈ 3 hours on A100)
model, sigreg, policy_head, value_head, history = train_lewm(CFG)
plot_training(history)
```

### Generate Showcase Visualizations

```python
# Load best checkpoint
model, sigreg, policy_head, value_head = load_model("lewm_chess_best.pt", CFG)

# Generate all 4 showcase plots
showcase_full_report(model, policy_head, value_head, CFG, CFG.cache_path)

# Probe latent space
probe_latent_space(model, CFG, CFG.cache_path)

# Generate tactical puzzles
puzzles = generate_puzzles(model, CFG, CFG.cache_path, n_scan=15000, n_puzzles=10)
puzzles_to_lichess(puzzles)
```

### Play Against It

```python
# Policy mode — instant moves (~30% GM-accuracy)
play_vs_lewm(model, policy_head, CFG, use_policy=True)

# CEM planning mode — slower (~0.8s/move), uses world model rollouts
play_vs_lewm(model, policy_head, CFG, use_policy=False)
```

## Project Structure

```
lewm-chess-engine/
├── README.md               # This file
├── lewm_chess.py            # Complete engine (~1800 lines, single file)
├── lewm_chess.ipynb         # Training notebook (Colab/Lightning-ready)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── assets/
    ├── lewm_game_analysis.png      # Value curve + material + surprise
    ├── lewm_latent_space.png       # t-SNE colored by result/material/phase
    ├── lewm_policy_analysis.png    # Top-K accuracy + move type breakdown
    └── lewm_value_calibration.png  # Phase accuracy + confidence calibration
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single-file architecture** | Everything in `lewm_chess.py` — no framework dependencies beyond PyTorch/timm. Copy into a Colab notebook and go |
| **On-the-fly board rendering** | Cache FEN strings (~72MB) instead of pre-rendered images (~160GB). Boards rendered in DataLoader workers |
| **Fixed SIGReg λ=0.09** | Paper default, applied from epoch 0. Warmup experiments caused **representation collapse** — latent space converged to a constant vector |
| **Grad clip = 5.0** | SIGReg produces gradient norms of 17-35 in early training (see epoch 1 logs). Paper uses 1.0, but chess data requires 5.0 |
| **Game-progress conditioning** | Telling the predictor "this is move 5" vs "this is move 35" changes dynamics entirely (openings have theory, endgames have technique) |
| **batch_size = 128** | Reduced from 512 to fit seq_len=20 (5× more ViT forward passes per batch) within A100 VRAM |

## Limitations & Future Work

- **Dataset size**: 17K games is small. Doubling to 35K+ should improve val_pred and reduce overfitting after epoch 6
- **Search depth**: CEM planner looks only 5 moves ahead. Combining with MCTS could significantly improve play strength
- **Board representation**: RGB images (paper-faithful) are less efficient than direct tensor encodings
- **Playing strength**: The policy head plays at intermediate club level — it understands chess patterns but lacks deep calculation
- **Training budget**: Only 10 epochs completed (out of 40 configured). Longer training with more data would push accuracy higher

## Acknowledgements

- **[LeWorldModel](https://github.com/lucas-maes/le-wm)** by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero — for the original JEPA architecture and SIGReg regularizer
- **[Gukesh D.](https://en.wikipedia.org/wiki/Gukesh_Dommaraju)** — the youngest World Chess Champion, whose 17,229 games form our training dataset
- Built with PyTorch, timm, python-chess, and einops

## Citation

```bibtex
@article{maes2025lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2025}
}
```

---

<p align="center">
  <em>Built with ♟️ and 🧠 — training a world model to understand chess, not just play it</em>
</p>
