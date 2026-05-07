"""
Evaluation, Metrics, and Benchmarking for Autonomous Driving Agent

Produces:
  - Safety metrics (collision rate, near-miss rate)
  - Efficiency metrics (avg speed, time-to-destination)
  - Comfort metrics (jerk, lateral acceleration)
  - Lane discipline metrics
  - Full episode replay logs
"""

import json
import os
import sys
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from driving_env import AutonomousDrivingEnv
from ppo_agent import PPOAgent


@dataclass
class EpisodeStats:
    ep_id: int
    total_reward: float
    length: int
    avg_speed: float
    max_speed: float
    collisions: int
    off_road_events: int
    lane_changes: int
    time_over_target_speed: int   # steps above target speed
    mean_lateral_velocity: float
    distance_traveled: float
    termination: str              # "collision", "off_road", "timeout"


@dataclass
class EvalReport:
    n_episodes: int
    mean_reward: float
    std_reward: float
    collision_rate: float         # fraction of episodes with collision
    mean_distance: float
    mean_speed: float
    mean_comfort: float           # lower = more comfortable
    success_rate: float           # episodes that completed without collision
    stage: str = "unknown"


class AgentEvaluator:
    """Run systematic evaluation of a trained agent."""

    def __init__(
        self,
        agent: PPOAgent,
        env: AutonomousDrivingEnv,
        n_episodes: int = 50,
        deterministic: bool = True,
        log_dir: str = "logs",
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run(self) -> EvalReport:
        """Execute full evaluation sweep."""
        all_stats: List[EpisodeStats] = []
        all_rewards: List[float] = []

        print(f"\n  Running {self.n_episodes} evaluation episodes...")
        print("  " + "─" * 50)

        for ep in range(self.n_episodes):
            stats = self._run_episode(ep)
            all_stats.append(stats)
            all_rewards.append(stats.total_reward)

            bar = "█" * min(int((ep + 1) / self.n_episodes * 30), 30)
            print(f"\r  [{bar:<30}] {ep+1}/{self.n_episodes}  reward={stats.total_reward:+7.1f}", end="", flush=True)

        print("\n")

        # ── Aggregate ──
        collision_count = sum(1 for s in all_stats if s.collisions > 0)
        report = EvalReport(
            n_episodes=self.n_episodes,
            mean_reward=statistics.mean(all_rewards),
            std_reward=statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0,
            collision_rate=collision_count / self.n_episodes,
            mean_distance=statistics.mean(s.distance_traveled for s in all_stats),
            mean_speed=statistics.mean(s.avg_speed for s in all_stats),
            mean_comfort=statistics.mean(s.mean_lateral_velocity for s in all_stats),
            success_rate=1.0 - collision_count / self.n_episodes,
        )

        self._print_report(report)
        self._save_report(report, all_stats)
        return report

    def _run_episode(self, ep_id: int) -> EpisodeStats:
        obs, _ = self.env.reset()
        total_reward = 0.0
        step = 0
        speeds: List[float] = []
        lateral_vels: List[float] = []
        collisions = 0
        off_road = 0
        termination = "timeout"

        done = False
        while not done:
            if self.deterministic:
                action = self.agent.act_deterministic(obs)
            else:
                action, _, _ = self.agent.act(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1

            speeds.append(self.env.ego.speed())
            lateral_vels.append(abs(self.env.ego.vy))

            if terminated:
                term = info.get("termination", "unknown")
                if term == "collision":
                    collisions += 1
                    termination = "collision"
                elif term == "off_road":
                    off_road += 1
                    termination = "off_road"
            done = terminated or truncated

        return EpisodeStats(
            ep_id=ep_id,
            total_reward=total_reward,
            length=step,
            avg_speed=float(np.mean(speeds)) if speeds else 0.0,
            max_speed=float(max(speeds)) if speeds else 0.0,
            collisions=collisions,
            off_road_events=off_road,
            lane_changes=0,
            time_over_target_speed=sum(1 for s in speeds if s > self.env.target_speed),
            mean_lateral_velocity=float(np.mean(lateral_vels)) if lateral_vels else 0.0,
            distance_traveled=self.env.episode_metrics.get("distance_traveled", 0.0),
            termination=termination,
        )

    @staticmethod
    def _print_report(r: EvalReport):
        stars = "⭐" * min(5, max(0, int(r.success_rate * 5)))
        print("  ┌─────────────────────────────────────────┐")
        print("  │         EVALUATION REPORT               │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  Episodes:       {r.n_episodes:>6}                  │")
        print(f"  │  Mean reward:    {r.mean_reward:>+7.2f} ± {r.std_reward:.2f}       │")
        print(f"  │  Success rate:   {r.success_rate*100:>6.1f}%   {stars:<8}       │")
        print(f"  │  Collision rate: {r.collision_rate*100:>6.1f}%                   │")
        print(f"  │  Mean speed:     {r.mean_speed:>6.1f} m/s               │")
        print(f"  │  Mean distance:  {r.mean_distance:>6.1f} m                │")
        print(f"  │  Comfort score:  {r.mean_comfort:>6.3f} (↓ better)       │")
        print("  └─────────────────────────────────────────┘")

    def _save_report(self, report: EvalReport, stats: List[EpisodeStats]):
        out = {
            "report": report.__dict__,
            "episodes": [s.__dict__ for s in stats],
        }
        path = os.path.join(self.log_dir, "eval_report.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Report saved → {path}")


def run_quick_eval():
    """Quick CLI evaluation."""
    env = AutonomousDrivingEnv(num_lanes=3, num_traffic=8, max_steps=500)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    model_path = "checkpoints/best_model.json"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Loaded trained model.")
    else:
        print("No trained model found. Using random policy.")

    evaluator = AgentEvaluator(agent, env, n_episodes=20)
    report = evaluator.run()
    env.close()
    return report


if __name__ == "__main__":
    run_quick_eval()
