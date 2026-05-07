import json
import matplotlib.pyplot as plt

# Load metrics
with open("logs/metrics.json") as f:
    data = json.load(f)

# Extract rewards (CORRECT KEY)
rewards = [m["mean_reward"] for m in data]

# Extract timesteps (optional but better graph)
timesteps = [m["timestep"] for m in data]

# Plot
plt.plot(timesteps, rewards)
plt.title("Training Reward Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.grid()
plt.show()