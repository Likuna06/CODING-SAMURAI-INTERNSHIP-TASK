"""
Advanced Reward Shaping & Curriculum Learning for Autonomous Driving RL

Features:
  - Multi-objective reward with tunable weights
  - Curriculum: difficulty ramps up as agent improves
  - Shaped rewards: TTC (time-to-collision), jerk penalty, smooth driving
  - Potential-based reward shaping (preserves optimal policy)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardWeights:
    """Configurable reward weights for multi-objective driving."""
    speed_tracking: float = 1.0       # reward for matching target speed
    lane_keeping:   float = 0.8       # reward for staying centered
    collision:      float = -30.0     # terminal penalty
    off_road:       float = -10.0     # terminal penalty
    forward:        float = 0.2       # reward for forward progress
    comfort:        float = -0.15     # penalty for lateral jerk
    safety_margin:  float = -0.5      # penalty for being too close to others
    overtaking:     float = 0.3       # reward for safely passing slower cars


class AdvancedRewardShaper:
    """
    Potential-based reward shaping that augments the base environment reward.
    Uses F(s, a, s') = γ·Φ(s') − Φ(s) to stay policy-invariant.
    """

    def __init__(self, weights: Optional[RewardWeights] = None, gamma: float = 0.99):
        self.w = weights or RewardWeights()
        self.gamma = gamma
        self._prev_potential: Optional[float] = None
        self._prev_speed: Optional[float] = None

    def reset(self):
        self._prev_potential = None
        self._prev_speed = None

    def potential(self, ego_speed: float, target_speed: float, dist_to_center: float,
                  lane_width: float) -> float:
        """Φ(s) = speed goodness + lane goodness."""
        speed_p = 1.0 - abs(ego_speed - target_speed) / target_speed
        lane_p = max(0.0, 1.0 - dist_to_center / (lane_width / 2))
        return speed_p + 0.5 * lane_p

    def compute_shaped_reward(
        self,
        ego_speed: float,
        target_speed: float,
        dist_center: float,
        lane_width: float,
        lateral_velocity: float,
        front_clearance: float,      # distance to nearest front vehicle
        base_reward: float,
        done: bool,
        info: dict,
    ) -> float:
        """Return base_reward + shaping terms."""

        r = base_reward

        # ── Time-to-collision penalty ──
        if ego_speed > 0 and front_clearance < 30.0:
            ttc = front_clearance / max(ego_speed, 0.1)
            if ttc < 2.0:
                r += self.w.safety_margin * (1.0 - ttc / 2.0)

        # ── Jerk penalty (abrupt lateral changes) ──
        if self._prev_speed is not None:
            speed_change = abs(ego_speed - self._prev_speed)
            r += self.w.comfort * speed_change / target_speed

        # ── Potential-based shaping ──
        phi = self.potential(ego_speed, target_speed, dist_center, lane_width)
        if self._prev_potential is not None:
            r += self.gamma * phi - self._prev_potential

        # Store for next step
        self._prev_potential = phi if not done else None
        self._prev_speed = ego_speed if not done else None

        return r


@dataclass
class CurriculumStage:
    name: str
    num_traffic: int
    max_speed: float
    target_speed: float
    sensor_range: float
    min_episodes: int          # before eligible to advance
    min_mean_reward: float     # threshold to advance


class CurriculumScheduler:
    """
    Progressive curriculum:
      Stage 0 — Empty road, slow speed (learn basic control)
      Stage 1 — Low traffic, medium speed
      Stage 2 — Medium traffic, target speed
      Stage 3 — Dense traffic, full speed
      Stage 4 — Dense + narrow sensors (more realistic)
    """

    STAGES = [
        CurriculumStage("Beginner",    num_traffic=0,  max_speed=15.0, target_speed=10.0, sensor_range=80.0, min_episodes=50,  min_mean_reward=150.0),
        CurriculumStage("Easy",        num_traffic=3,  max_speed=20.0, target_speed=14.0, sensor_range=60.0, min_episodes=100, min_mean_reward=200.0),
        CurriculumStage("Medium",      num_traffic=6,  max_speed=25.0, target_speed=18.0, sensor_range=50.0, min_episodes=150, min_mean_reward=250.0),
        CurriculumStage("Hard",        num_traffic=10, max_speed=30.0, target_speed=22.0, sensor_range=40.0, min_episodes=200, min_mean_reward=280.0),
        CurriculumStage("Expert",      num_traffic=15, max_speed=35.0, target_speed=25.0, sensor_range=30.0, min_episodes=300, min_mean_reward=300.0),
    ]

    def __init__(self):
        self.stage_idx = 0
        self.episode_count = 0
        self.recent_rewards: list = []

    @property
    def current_stage(self) -> CurriculumStage:
        return self.STAGES[self.stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    def record_episode(self, reward: float) -> bool:
        """
        Record episode reward. Returns True if stage advanced.
        """
        self.episode_count += 1
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)

        if self.stage_idx < len(self.STAGES) - 1:
            stage = self.current_stage
            if (self.episode_count >= stage.min_episodes and
                    len(self.recent_rewards) >= 10 and
                    np.mean(self.recent_rewards) >= stage.min_mean_reward):
                self.stage_idx += 1
                self.episode_count = 0
                self.recent_rewards.clear()
                print(f"\n  🎓 Curriculum advanced → {self.current_stage.name}")
                return True
        return False

    def env_kwargs(self) -> dict:
        s = self.current_stage
        return {
            "num_traffic": s.num_traffic,
            "max_speed": s.max_speed,
            "target_speed": s.target_speed,
            "sensor_range": s.sensor_range,
        }

    def status(self) -> dict:
        return {
            "stage": self.stage_idx,
            "stage_name": self.stage_name,
            "episode_count": self.episode_count,
            "mean_recent_reward": float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
            "next_threshold": self.current_stage.min_mean_reward,
        }
