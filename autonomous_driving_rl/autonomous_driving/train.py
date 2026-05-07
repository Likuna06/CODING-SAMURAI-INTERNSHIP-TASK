"""
Training entry point for the Autonomous Driving PPO Agent.

Usage:
    python train.py                         # default settings
    python train.py --timesteps 1000000     # more training
    python train.py --eval                  # evaluate saved model
    python train.py --render                # render after training
"""

import argparse
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from driving_env import AutonomousDrivingEnv
from ppo_agent import PPOAgent, PPOConfig


def train(args):
    print("\n🚗  Autonomous Driving RL Agent — Training")
    print("=" * 60)

    # ── Config ──────────────────────────────────────────
    env_kwargs = dict(
        num_lanes=3,
        road_length=300.0,
        num_traffic=8,
        max_speed=30.0,
        target_speed=20.0,
        dt=0.1,
        max_steps=1000,
        sensor_range=50.0,
        num_lidar_beams=12,
        seed=args.seed,
    )

    ppo_config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.02,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        target_kl=0.015,
    )

    env = AutonomousDrivingEnv(**env_kwargs)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"  Env: {obs_dim}D obs, {action_dim}D action")
    print(f"  Road: {env_kwargs['num_lanes']} lanes, {env_kwargs['num_traffic']} traffic vehicles")
    print(f"  Training for {args.timesteps:,} timesteps\n")

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=ppo_config,
        checkpoint_dir="checkpoints",
        log_dir="logs",
    )

    def log_callback(metrics):
        """Write live metrics for dashboard."""
        with open("logs/live_metrics.json", "w") as f:
            json.dump(metrics, f,default=lambda x: float(x))

    agent.train(env, total_timesteps=args.timesteps, callback=log_callback)
    env.close()
    print("\n✅  Training complete!")


def evaluate(args):
    print("\n🔍  Evaluating saved agent...")
    env = AutonomousDrivingEnv(num_lanes=3, num_traffic=8, max_steps=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)

    model_path = "checkpoints/best_model.json"
    if not os.path.exists(model_path):
        print(f"  No model found at {model_path}. Train first.")
        return

    agent.load(model_path)
    print(f"  Loaded model: {model_path}")

    total_rewards = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1:2d}: reward = {ep_reward:+8.2f}")

    import statistics
    print(f"\n  Mean reward: {statistics.mean(total_rewards):.2f}")
    print(f"  Std reward:  {statistics.stdev(total_rewards):.2f}")
    env.close()


def demo_run(args):
    """Run one episode and save state log for dashboard demo."""
    print("\n🎮  Running demo episode...")
    env = AutonomousDrivingEnv(
        num_lanes=3,
        num_traffic=8,
        max_steps=300,
        render_mode="json",
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)

    model_path = "checkpoints/best_model.json"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("  Using trained model.")
    else:
        print("  No model found — running with random policy.")

    obs, _ = env.reset()
    frames = []
    done = False

    while not done:
        if os.path.exists(model_path):
            action = agent.act_deterministic(obs)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        done = terminated or truncated

    os.makedirs("logs", exist_ok=True)
    with open("logs/demo_frames.json", "w") as f:
        json.dump(frames, f)

    print(f"  Saved {len(frames)} frames → logs/demo_frames.json")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Driving RL Agent")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.eval:
        evaluate(args)
    elif args.demo:
        demo_run(args)
    else:
        train(args)
