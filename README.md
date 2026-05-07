# 🚗 Autonomous Driving RL Agent
### Proximal Policy Optimization (PPO) on a Custom Gym Environment

A full reinforcement learning pipeline for autonomous driving, built from scratch.
Trains an agent to navigate a multi-lane highway with traffic using lidar-like sensors.

---

## Project Structure

```
autonomous_driving/
├── src/
│   ├── driving_env.py       ← Custom Gym environment (Road, Vehicles, Sensors)
│   ├── ppo_agent.py         ← Full PPO agent + Actor-Critic network (NumPy)
│   ├── reward_shaping.py    ← Advanced reward shaping + curriculum learning
│   └── evaluate.py          ← Evaluation + metrics reporting
├── train.py                 ← Main training entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train (default: 200k timesteps)
python train.py

# Train longer
python train.py --timesteps 1000000

# Evaluate saved model
python train.py --eval

# Run a demo episode and save frames
python train.py --demo
```

---

## Environment: `AutonomousDrivingEnv`

### Road
- 3-lane highway, 300m long
- Configurable lane width, number of lanes, traffic density

### Observation Space (26-dim continuous)
| Component | Dims | Description |
|-----------|------|-------------|
| Ego state | 6 | position, velocity, heading, speed (normalised) |
| Lidar     | 12  | 12 radial beams → distance to nearest obstacle |
| Lane info | 4   | lane ID, dist-to-center, left/right margins |
| Traffic   | 4   | nearest front/rear vehicle distance & speed |

### Action Space (2-dim continuous, [-1, 1])
| Channel  | Meaning |
|----------|---------|
| Steering | Lateral acceleration (−1=left, +1=right) |
| Throttle | Longitudinal acceleration (−1=brake, +1=accelerate) |

### Reward Function
| Component | Weight | Description |
|-----------|--------|-------------|
| Speed tracking | +1.0 | Penalty for deviation from target speed |
| Lane-keeping   | +0.5 | Reward for staying centred in lane |
| Collision      | −20  | Terminal penalty |
| Off-road       | −5   | Terminal penalty |
| Forward progress| +0.1 | Reward per m/s of forward velocity |
| Comfort        | −0.1 | Penalty for lateral velocity |

---

## Algorithm: PPO (Proximal Policy Optimization)

### Architecture
```
Observation (26) 
    → Shared FC [256 → 128]
         ├── Actor head:  [128 → 64] → mean (2) + log_std
         └── Critic head: [128 → 64] → value (1)
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 (Adam) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy bonus | 0.02 |
| Rollout steps | 2048 |
| PPO epochs | 10 |
| Batch size | 64 |

### Training Loop
1. Collect N rollout steps using current policy
2. Compute GAE advantages and discounted returns
3. Run K epochs of mini-batch PPO updates
4. Early stop if KL divergence exceeds threshold
5. Save checkpoint if mean reward improved

---

## Curriculum Learning

The agent trains progressively:

| Stage | Traffic | Max Speed | Sensor Range |
|-------|---------|-----------|-------------|
| Beginner | 0 vehicles | 15 m/s | 80 m |
| Easy | 3 vehicles | 20 m/s | 60 m |
| Medium | 6 vehicles | 25 m/s | 50 m |
| Hard | 10 vehicles | 30 m/s | 40 m |
| Expert | 15 vehicles | 35 m/s | 30 m |

The agent advances to the next stage only when it achieves a mean episode reward above the threshold over the last 20 episodes.

---

## Advanced Reward Shaping

The `AdvancedRewardShaper` adds potential-based shaping on top of the base reward:

- **Time-to-collision (TTC)** penalty: penalises driving toward a vehicle with TTC < 2s
- **Jerk penalty**: penalises sudden speed changes for driving comfort
- **Potential-based shaping**: F(s,s') = γΦ(s') − Φ(s), guaranteed to preserve optimal policy

---

## Evaluation Metrics

Run `python train.py --eval` to get:

| Metric | Description |
|--------|-------------|
| Mean reward | Average return over 50 evaluation episodes |
| Success rate | % of episodes without collision |
| Collision rate | % of episodes with at least one collision |
| Mean speed | Average ego speed (m/s) |
| Mean distance | Average distance traveled per episode |
| Comfort score | Mean lateral velocity (lower = smoother) |

---

## Extending the Project

### Add PyTorch (Recommended for Production)
Replace `src/ppo_agent.py` neural network with PyTorch:
```python
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
```

### Use CARLA Simulator
Replace `AutonomousDrivingEnv` with a CARLA wrapper:
```python
import carla
# Connect to CARLA server, wrap in Gym interface
```

### Multi-Agent Training
Spawn multiple ego vehicles and use shared policy with independent observations.

---

## References
- [PPO Paper](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [GAE Paper](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015
- [OpenAI Gym](https://gymnasium.farama.org/)
- [CARLA Simulator](https://carla.org/)
- [Reinforcement Learning for Self-Driving Cars](https://youtu.be/Cy155O5R1Oo)
