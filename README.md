# Google Research Football — Kaggle Competition

## Overview

The [Google Research Football with Manchester City F.C.](https://www.kaggle.com/competitions/google-football) competition (2020) challenged participants to build AI agents that play 11v11 simulated football. Agents submit a Python function receiving game observations (player positions, ball state, game mode) and returning actions. Agents compete head-to-head on Kaggle's evaluation servers with Elo-style rating.

This was one of my highest competition finishes, with the "Marauding Wingers" agent (33 upvotes on Kaggle).

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Rule-Based Foundation: Domain Expertise as Code

The core strategy translated real football tactics into code — specifically the "marauding wingers" formation: wide players sprint down the flanks and deliver crosses into the box. This gave a strong baseline without any machine learning.

### 2. Zone-Based Decision Architecture

The field is divided into zones (defensive third, wing corridors, crossing range, shooting range) with different behaviors per zone. Set pieces (corners, free kicks, goal kicks, penalties) each have dedicated handlers with positional logic.

### 3. Opponent-Aware Mechanics

An `opp_prox` function checks opponent density in configurable proximity windows around the controlled player, enabling context-sensitive decisions — sprint in open space, dribble under pressure, pass when crowded, shoot when clear.

### 4. Goalkeeper Exploitation

Specific logic detects when the opposing goalkeeper is out of position (too far from goal line or off-center) and triggers long-range shots — a high-value heuristic that rule-based opponents rarely defend against.

### 5. Sprint/Dribble State Machine

The final agent manages sprint vs. dribble mode based on field position and opponent proximity — sprinting in the midfield when clear, dribbling in the attacking third to maintain control, with explicit sticky-action management for the environment's action queue.

### 6. Imitation Learning Experiment

Attempted to replace midfield build-up rules with a RandomForest classifier trained on community replay data (features: distance and heading to all 22 players). The hybrid agent kept rules for defense and set pieces while using ML for offensive movement decisions.

## Repository Structure

```
└── notebooks/
    ├── 01-baseline-agent.ipynb                # Competition starter — chase-ball-and-shoot baseline
    ├── 02-marauding-wingers-dev.ipynb          # Core development notebook (14 versions of iteration)
    ├── 03-marauding-wingers-published.ipynb    # ★ Published agent at score 1041 (33 upvotes)
    ├── 04-injury-time-final-agent.ipynb        # Most advanced rule-based agent
    └── 05-imitation-learning-hybrid.ipynb      # ML experiment — RandomForest imitation learning
```

## Tech Stack

- **Agent Framework**: Google Research Football Environment (GRF)
- **Rule Engine**: Custom zone-based decision tree with set piece handlers
- **ML Experiment**: scikit-learn RandomForest (imitation learning from replay data)
- **Infrastructure**: Kaggle Notebooks, Elo-rated agent evaluation

## Competition

- **Name**: [Google Research Football with Manchester City F.C.](https://www.kaggle.com/competitions/google-football)
- **Type**: Reinforcement Learning / Agent Competition
- **Metric**: Elo rating from head-to-head matches
- **Timeline**: August — December 2020
