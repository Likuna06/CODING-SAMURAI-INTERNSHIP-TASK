import streamlit as st
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # ✅ for drawing cars

st.set_page_config(page_title="Autonomous Driving", layout="centered")

st.title("🚗 Autonomous Driving Simulation")

# Load frames
try:
    with open("logs/demo_frames.json") as f:
        frames = json.load(f)
except:
    st.error("Run demo first: python train.py --demo")
    st.stop()

# Filter out bad frames
frames = [f for f in frames if f is not None]

if len(frames) == 0:
    st.error("No valid frames found. Run demo again.")
    st.stop()

# Animation placeholder
frame_display = st.empty()

# Controls
col1, col2 = st.columns(2)
play = col1.button("▶️ Play Simulation")
speed = col2.slider("Speed", 0.01, 0.2, 0.05)

# Play simulation
if play:
    for frame in frames:

        fig, ax = plt.subplots()

        road = frame["road"]
        ego = frame["ego"]
        traffic = frame["traffic"]

        # 🛣️ Draw lanes
        for i in range(road["num_lanes"] + 1):
            y = i * road["lane_width"]
            ax.axhline(y, linestyle='--', color='gray')

        # 🚗 Ego car (RED rectangle)
        ego_car = patches.Rectangle(
            (ego["x"] - 2, ego["y"] - 1),
            4, 2,
            color='red'
        )
        ax.add_patch(ego_car)

        # 🚙 Traffic cars (BLUE rectangles)
        for t in traffic:
            car = patches.Rectangle(
                (t["x"] - 2, t["y"] - 1),
                4, 2,
                color='blue'
            )
            ax.add_patch(car)

        # 📐 Camera follows ego
        ax.set_xlim(ego["x"] - 50, ego["x"] + 50)
        ax.set_ylim(0, road["total_width"])

        # Labels
        ax.set_title(f"Step: {frame.get('step', 0)}")
        ax.set_xlabel("Road Length")
        ax.set_ylabel("Lane Position")

        frame_display.pyplot(fig)
        plt.close(fig)  # ✅ FIX memory warning
        time.sleep(speed)

st.success("Simulation ready 🚀")