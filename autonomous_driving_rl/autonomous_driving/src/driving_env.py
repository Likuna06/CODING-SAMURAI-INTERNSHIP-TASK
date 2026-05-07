"""
Autonomous Driving Environment - Custom OpenAI Gym-compatible environment
Simulates a 2D top-down driving scenario with:
- Multi-lane highway with traffic
- Lidar-like distance sensors
- Speed/acceleration control
- Collision detection
- Lane-keeping rewards
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
import matplotlib.patches as patches

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Vehicle:
    x: float
    y: float
    vx: float          # velocity x
    vy: float          # velocity y
    width: float = 2.0
    length: float = 4.0
    color: str = "blue"

    def corners(self) -> List[Tuple[float, float]]:
        """Return 4 corners of the vehicle rectangle."""
        hw, hl = self.width / 2, self.length / 2
        angle = math.atan2(self.vy, self.vx) if (self.vx != 0 or self.vy != 0) else 0
        corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        rotated = []
        for cx, cy in corners:
            rx = cx * math.cos(angle) - cy * math.sin(angle) + self.x
            ry = cx * math.sin(angle) + cy * math.cos(angle) + self.y
            rotated.append((rx, ry))
        return rotated

    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    def heading(self) -> float:
        return math.atan2(self.vy, self.vx)


@dataclass
class Road:
    num_lanes: int = 3
    lane_width: float = 4.0
    length: float = 200.0

    @property
    def total_width(self) -> float:
        return self.num_lanes * self.lane_width

    def lane_center(self, lane: int) -> float:
        """Y center of given lane (0-indexed)."""
        return (lane + 0.5) * self.lane_width

    def which_lane(self, y: float) -> int:
        """Return lane index for given y position."""
        lane = int(y / self.lane_width)
        return max(0, min(self.num_lanes - 1, lane))

    def distance_to_lane_center(self, y: float, lane: int) -> float:
        return abs(y - self.lane_center(lane))


# ─────────────────────────────────────────────
# The Environment
# ─────────────────────────────────────────────

class AutonomousDrivingEnv(gym.Env):
    """
    Custom 2D top-down autonomous driving environment.

    Observation space (26-dim):
        - ego: [x_norm, y_norm, vx_norm, vy_norm, heading_norm, speed_norm]  (6)
        - lidar sensors: 12 beams (distances to nearest obstacle)            (12)
        - lane info: [lane_id/num_lanes, dist_to_center, dist_left, dist_right] (4)
        - traffic: [closest_front_dist, closest_front_speed,
                    closest_rear_dist, closest_rear_speed]                   (4)

    Action space (continuous, 2-dim):
        - steering: [-1, 1] → maps to lateral velocity change
        - throttle:  [-1, 1] → maps to longitudinal acceleration/braking
    """

    metadata = {"render_modes": ["human", "rgb_array", "json"]}

    def __init__(
        self,
        num_lanes: int = 3,
        road_length: float = 300.0,
        num_traffic: int = 8,
        max_speed: float = 30.0,        # m/s (~108 km/h)
        target_speed: float = 20.0,     # m/s (~72 km/h)
        dt: float = 0.1,                # simulation timestep (s)
        max_steps: int = 1000,
        sensor_range: float = 50.0,
        num_lidar_beams: int = 12,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.road = Road(num_lanes=num_lanes, length=road_length)
        self.num_traffic = num_traffic
        self.max_speed = max_speed
        self.target_speed = target_speed
        self.dt = dt
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.num_lidar_beams = num_lidar_beams
        self.render_mode = render_mode

        # Spaces
        obs_dim = 6 + num_lidar_beams + 4 + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # State
        self.ego: Optional[Vehicle] = None
        self.traffic: List[Vehicle] = []
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_metrics: Dict = {}

        # RNG
        self._rng = np.random.default_rng(seed)

        # Rendering
        self._render_surface = None

    # ── Reset ──────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.total_reward = 0.0

        # Spawn ego in middle lane
        start_lane = self.road.num_lanes // 2
        self.ego = Vehicle(
            x=30.0,
            y=self.road.lane_center(start_lane),
            vx=self.target_speed * 0.8,
            vy=0.0,
            color="red",
        )

        # Spawn traffic
        self.traffic = []
        for i in range(self.num_traffic):
            lane = int(self._rng.integers(0, self.road.num_lanes))
            while True:
                dist = self._rng.uniform(20, self.road.length - 20)
                if abs(dist - self.ego.x) > 30:  # keep distance
                    break
            speed = self._rng.uniform(10, self.max_speed * 0.8)
            self.traffic.append(Vehicle(
                x=dist,
                y=self.road.lane_center(lane),
                vx=speed,
                vy=0.0,
                color="blue",
            ))

        self.episode_metrics = {
            "collisions": 0,
            "lane_changes": 0,
            "avg_speed": [],
            "distance_traveled": 0.0,
        }

        return self._get_obs(), self._get_info()

    # ── Step ───────────────────────────────────

    def step(self, action: np.ndarray):
        assert self.ego is not None, "Call reset() first."

        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        # ── Physics ──
        max_accel = 4.0    # m/s²
        max_steer = 3.0    # lateral m/s² equivalent

        # Apply throttle → longitudinal
        ax = throttle * max_accel
        # Apply steering → lateral (simplified kinematic)
        ay = steering * max_steer

        # Update velocity (with drag)
        drag = 0.05
        self.ego.vx = np.clip(self.ego.vx + ax * self.dt - drag * self.ego.vx,
                               0.0, self.max_speed)
        self.ego.vy = np.clip(self.ego.vy + ay * self.dt - drag * self.ego.vy,
                               -self.max_speed * 0.3, self.max_speed * 0.3)

        # Update position
        prev_x = self.ego.x
        self.ego.x += self.ego.vx * self.dt
        self.ego.y += self.ego.vy * self.dt

        # Clamp to road
        self.ego.y = np.clip(self.ego.y, 0.5, self.road.total_width - 0.5)

        # Update traffic (simple constant speed + lane keeping)
        for t in self.traffic:
            t.x += t.vx * self.dt
            # Wrap around
            if t.x > self.road.length:
                t.x = 0.0
                t.y = self.road.lane_center(int(self._rng.integers(0, self.road.num_lanes)))
                t.vx = self._rng.uniform(10, self.max_speed * 0.8)
            # Gentle lane centering
            target_y = self.road.lane_center(self.road.which_lane(t.y))
            t.y += (target_y - t.y) * 0.05

        # ── Metrics ──
        self.episode_metrics["avg_speed"].append(self.ego.speed())
        self.episode_metrics["distance_traveled"] += self.ego.x - prev_x

        # ── Reward ──
        reward, done, info = self._compute_reward()
        self.total_reward += reward
        self.step_count += 1

        truncated = self.step_count >= self.max_steps
        if truncated or done:
            info["episode"] = {
                "r": self.total_reward,
                "l": self.step_count,
                "avg_speed": float(np.mean(self.episode_metrics["avg_speed"])),
                "distance": self.episode_metrics["distance_traveled"],
                "collisions": self.episode_metrics["collisions"],
            }

        obs = self._get_obs()
        return obs, reward, done, truncated, info

    # ── Reward ─────────────────────────────────

    def _compute_reward(self) -> Tuple[float, bool, dict]:
        reward = 0.0
        done = False
        info = {}

        # 1. Speed reward — encourage target speed
        speed = self.ego.speed()
        speed_err = abs(speed - self.target_speed) / self.target_speed
        reward += 1.0 * (1.0 - speed_err)

        # 2. Lane-keeping reward
        lane = self.road.which_lane(self.ego.y)
        dist_center = self.road.distance_to_lane_center(self.ego.y, lane)
        lane_reward = max(0.0, 1.0 - dist_center / (self.road.lane_width / 2))
        reward += 0.5 * lane_reward

        # 3. Collision penalty
        collision = self._check_collision()
        if collision:
            reward -= 20.0
            done = True
            self.episode_metrics["collisions"] += 1
            info["termination"] = "collision"

        # 4. Road boundary penalty
        margin = 0.3
        if self.ego.y < margin or self.ego.y > self.road.total_width - margin:
            reward -= 5.0
            done = True
            info["termination"] = "off_road"

        # 5. Forward progress bonus
        reward += 0.1 * (self.ego.vx / self.max_speed)

        # 6. Comfort — penalise large lateral velocity
        reward -= 0.1 * abs(self.ego.vy / self.max_speed)

        return reward, done, info

    # ── Collision Detection ────────────────────

    def _check_collision(self) -> bool:
        for t in self.traffic:
            dx = abs(self.ego.x - t.x)
            dy = abs(self.ego.y - t.y)
            if dx < (self.ego.length + t.length) / 2 and dy < (self.ego.width + t.width) / 2:
                return True
        return False

    # ── Lidar ─────────────────────────────────

    def _lidar_scan(self) -> np.ndarray:
        """Cast beams from ego in all directions, return normalised distances."""
        distances = np.ones(self.num_lidar_beams)  # 1.0 = max range (normalised)
        angle_step = 2 * math.pi / self.num_lidar_beams

        for beam_idx in range(self.num_lidar_beams):
            angle = beam_idx * angle_step
            dx = math.cos(angle)
            dy = math.sin(angle)

            min_dist = self.sensor_range

            # Check all traffic vehicles
            for t in self.traffic:
                # Simple AABB ray-box intersection
                for step in np.linspace(0, self.sensor_range, 50):
                    rx = self.ego.x + dx * step
                    ry = self.ego.y + dy * step
                    if (abs(rx - t.x) < t.length / 2 and
                            abs(ry - t.y) < t.width / 2):
                        min_dist = min(min_dist, step)
                        break

            # Check road boundaries
            for step in np.linspace(0, self.sensor_range, 50):
                ry = self.ego.y + dy * step
                if ry <= 0 or ry >= self.road.total_width:
                    min_dist = min(min_dist, step)
                    break

            distances[beam_idx] = min_dist / self.sensor_range

        return distances

    # ── Observation ───────────────────────────

    def _get_obs(self) -> np.ndarray:
        ego = self.ego

        # Ego state (normalised)
        ego_obs = np.array([
            ego.x / self.road.length,
            ego.y / self.road.total_width,
            ego.vx / self.max_speed,
            ego.vy / self.max_speed,
            math.atan2(ego.vy, ego.vx) / math.pi,
            ego.speed() / self.max_speed,
        ], dtype=np.float32)

        # Lidar
        lidar = self._lidar_scan().astype(np.float32)

        # Lane info
        lane = self.road.which_lane(ego.y)
        dist_center = self.road.distance_to_lane_center(ego.y, lane)
        dist_left = ego.y
        dist_right = self.road.total_width - ego.y
        lane_obs = np.array([
            lane / self.road.num_lanes,
            dist_center / (self.road.lane_width / 2),
            dist_left / self.road.total_width,
            dist_right / self.road.total_width,
        ], dtype=np.float32)

        # Closest front/rear traffic
        front_vehicles = [(t.x - ego.x, t.vx) for t in self.traffic if t.x > ego.x]
        rear_vehicles = [(ego.x - t.x, t.vx) for t in self.traffic if t.x <= ego.x]
        front_dist, front_spd = min(front_vehicles, key=lambda v: v[0], default=(self.sensor_range, 0.0))
        rear_dist, rear_spd = min(rear_vehicles, key=lambda v: v[0], default=(self.sensor_range, 0.0))
        traffic_obs = np.array([
            front_dist / self.sensor_range,
            front_spd / self.max_speed,
            rear_dist / self.sensor_range,
            rear_spd / self.max_speed,
        ], dtype=np.float32)

        obs = np.concatenate([ego_obs, lidar, lane_obs, traffic_obs])
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "ego_x": self.ego.x,
            "ego_y": self.ego.y,
            "ego_speed": self.ego.speed(),
            "ego_lane": self.road.which_lane(self.ego.y),
            "step": self.step_count,
        }

    # ── Render (JSON for dashboard) ────────────
    def render(self):
    # ✅ If Streamlit/demo mode → return JSON
        if self.render_mode == "json":
            return self._render_json()

    # 🎨 Otherwise → show matplotlib animation (your existing code)
        plt.clf()

    # lanes
        for i in range(self.road.num_lanes + 1):
            y = i * self.road.lane_width
            plt.axhline(y, linestyle='--', color='gray')

    # 🚗 ego car
        plt.text(self.ego.x, self.ego.y, "🚗", fontsize=20,
                ha='center', va='center')

    # 🚙 traffic cars
        for t in self.traffic:
            plt.text(t.x, t.y, "🚙", fontsize=16,
                    ha='center', va='center')

        plt.xlim(0, self.road.length)
        plt.ylim(0, self.road.total_width)

        plt.title(f"Step: {self.step_count}")
        plt.pause(0.05)

        return None
    def _render_json(self) -> dict:
        """Return state as JSON for the web dashboard."""
        return {
            "road": {
                "num_lanes": self.road.num_lanes,
                "lane_width": self.road.lane_width,
                "length": self.road.length,
                "total_width": self.road.total_width,
            },
            "ego": {
                "x": round(self.ego.x, 2),
                "y": round(self.ego.y, 2),
                "vx": round(self.ego.vx, 2),
                "vy": round(self.ego.vy, 2),
                "speed": round(self.ego.speed(), 2),
                "lane": self.road.which_lane(self.ego.y),
            },
            "traffic": [
                {
                    "x": round(t.x, 2),
                    "y": round(t.y, 2),
                    "vx": round(t.vx, 2),
                    "speed": round(t.speed(), 2),
                }
                for t in self.traffic
            ],
            "step": self.step_count,
            "total_reward": round(self.total_reward, 2),
        }

    def close(self):
        pass
