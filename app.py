# app.py
import time
import math
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------- Basic Styles -------------
st.set_page_config(page_title="A4S+ Prototype", layout="wide")
st.markdown("""
<style>
.small {font-size:0.9rem;color:#666;}
h1, h2, h3 { margin-bottom: .4rem; }
.block {padding: .6rem .8rem; background:#0f172a0d; border:1px solid #e2e8f0; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.title("A4S+ — Prototype (Software-only)")

tabs = st.tabs([
    "1) Soldier Health Alerts",
    "2) Safe Route Planner (A*)",
    "3) Drone Swarm Viz",
    "4) Security Monitor"
])

# =====================================================================
# 1) Soldier Health Alerts (Digital Twin — simplified rules)
# =====================================================================
with tabs[0]:
    st.subheader("Soldier Health Alerts (Digital Twin — simplified)")
    st.markdown("<div class='small'>CSV-driven demo. Simple rules flag risk.</div>", unsafe_allow_html=True)

    # Load CSV
    try:
        df = pd.read_csv("soldier_vitals.csv")
    except Exception:
        st.error("Couldn't find soldier_vitals.csv. Create it in the same folder.")
        st.stop()

    # Risk scoring (simple, explainable)
    def risk_score(row):
        score = 0
        reasons = []
        if row["spo2"] < 60: 
            score += 50; reasons.append("Very low SpO2")
        elif row["spo2"] < 70: 
            score += 25; reasons.append("Low SpO2")
        if row["heart_rate"] > 115: 
            score += 20; reasons.append("High heart rate")
        if row["skin_temp_c"] <= -12: 
            score += 20; reasons.append("Extreme cold exposure")
        if row["altitude_m"] >= 5000: 
            score += 10; reasons.append("Very high altitude")
        return score, reasons

    scores, reasons_all = [], []
    for _, r in df.iterrows():
        s, rs = risk_score(r)
        scores.append(s)
        reasons_all.append(", ".join(rs))
    df["risk_score"] = scores
    df["reasons"] = reasons_all
    df["risk_level"] = pd.cut(df["risk_score"], bins=[-1,19,49,1000], labels=["Low","Medium","High"])

    # Show table with conditional highlight
    def color_risk(val):
        if val == "High": return "background-color: #fecaca"
        if val == "Medium": return "background-color: #fde68a"
        return "background-color: #bbf7d0"
    st.dataframe(df.style.applymap(color_risk, subset=["risk_level"]))

    # Quick alert feed
    high_risk = df[df["risk_level"] == "High"]
    if not high_risk.empty:
        st.error(f"ALERT: {len(high_risk)} soldier(s) at HIGH risk → {', '.join(high_risk['name'].tolist())}")
    else:
        st.success("No HIGH risk soldiers at the moment.")

    st.markdown("**Explanation sample**")
    iid = st.selectbox("Pick soldier", df["id"])
    row = df[df["id"] == iid].iloc[0]
    sc, rs = risk_score(row)
    st.write(f"Risk score: {sc} | Reasons: {', '.join(rs) if rs else '—'}")

# =====================================================================
# 2) Safe Route Planner (A* on a grid with danger zones)
# =====================================================================
with tabs[1]:
    st.subheader("Safe Route Planner (A* over risk-weighted grid)")
    st.markdown("<div class='small'>Grid = terrain. Red cells = danger (avalanche/crevasse). A* avoids them.</div>", unsafe_allow_html=True)

    colA, colB = st.columns([1,1])
    with colA:
        grid_size = st.slider("Grid size (N x N)", 10, 40, 20)
        danger_ratio = st.slider("Danger density", 0.00, 0.45, 0.25, 0.01)
        rng_seed = st.number_input("Random seed", 0, 9999, 42)
        start = (0, 0)
        goal = (grid_size - 1, grid_size - 1)

    random.seed(rng_seed); np.random.seed(rng_seed)

    # Build risk grid
    risk = np.zeros((grid_size, grid_size), dtype=float)
    mask = np.random.rand(grid_size, grid_size) < danger_ratio
    risk[mask] = 100.0  # impassable/high-risk
    risk[start] = 0; risk[goal] = 0

    # A* implementation
    from heapq import heappush, heappop
    def neighbors(r, c, n):
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < n and 0 <= cc < n:
                yield rr, cc
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def a_star(risk_grid, start, goal):
        n = risk_grid.shape[0]
        open_set = []
        heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
        came_from = {}
        gscore = {start: 0}
        while open_set:
            f, g, node, parent = heappop(open_set)
            if node in came_from: continue
            came_from[node] = parent
            if node == goal:
                path = []
                cur = node
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                return path[::-1]
            r, c = node
            for rr, cc in neighbors(r, c, n):
                w = risk_grid[rr, cc]
                if w >= 100: continue
                ng = g + (1 + w)
                if (rr, cc) not in gscore or ng < gscore[(rr, cc)]:
                    gscore[(rr, cc)] = ng
                    heappush(open_set, (ng + heuristic((rr, cc), goal), ng, (rr, cc), node))
        return None

    path = a_star(risk, start, goal)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(risk, cmap="Reds", origin="upper")
    ax.set_title("Risk Map (red = danger) + A* path (green)")
    ax.set_xticks([]); ax.set_yticks([])
    if path:
        ys = [p[0] for p in path]; xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2.5)
        ax.scatter([start[1], goal[1]], [start[0], goal[0]], s=60)
        st.success(f"Path found with {len(path)} steps.")
    else:
        st.error("No safe path found — too many danger cells.")
    st.pyplot(fig)

# =====================================================================
# 3) Drone Swarm Visualization
# =====================================================================
with tabs[2]:
    st.subheader("Drone Swarm Visualization (simple sweep + reassignment)")
    st.markdown("<div class='small'>Dots = drones. They sweep rows. Kill one → others take over.</div>", unsafe_allow_html=True)

    N = st.slider("Grid width/height", 10, 40, 20, key="swarmN")
    num_drones = st.slider("Number of drones", 2, 8, 4)
    steps_per_click = st.slider("Steps per tick", 1, 10, 3)

    if "drones" not in st.session_state:
        st.session_state.drones = [{"x": 0, "y": i*(N//num_drones), "alive": True} for i in range(num_drones)]
        st.session_state.cover = np.zeros((N, N), dtype=np.int32)

    cols = st.columns([1,1,1,1])
    with cols[0]:
        if st.button("Step"):
            for _ in range(steps_per_click):
                for d in st.session_state.drones:
                    if not d["alive"]: continue
                    st.session_state.cover[d["y"], d["x"]] = 1
                    if d["x"] < N-1:
                        d["x"] += 1
                    else:
                        next_rows = [r for r in range(N) if st.session_state.cover[r].sum() < N]
                        if next_rows:
                            d["x"] = 0; d["y"] = min(next_rows)
                        else:
                            d["x"], d["y"] = 0, 0
    with cols[1]:
        if st.button("Kill a drone"):
            alive = [d for d in st.session_state.drones if d["alive"]]
            if alive: random.choice(alive)["alive"] = False
    with cols[2]:
        if st.button("Respawn all"):
            for d in st.session_state.drones: d["alive"] = True
    with cols[3]:
        if st.button("Reset world"):
            st.session_state.pop("drones", None)
            st.session_state.pop("cover", None)
            st.experimental_rerun()

    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(st.session_state.cover, origin="upper")
    xs = [d["x"] for d in st.session_state.drones if d["alive"]]
    ys = [d["y"] for d in st.session_state.drones if d["alive"]]
    ax2.scatter(xs, ys, s=80)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Coverage (heat) + Drone positions (dots)")
    st.pyplot(fig2)

# =====================================================================
# 4) Security Monitor (new tab)
# =====================================================================
with tabs[3]:
    st.subheader("Security Monitor (Prototype)")
    st.markdown("<div class='small'>Simulated cyber threats: jamming, spoofing, login attempts.</div>", unsafe_allow_html=True)

    # Simulated events
    events = [
        {"Time": "12:01", "Event": "Drone Bravo jammed", "Action": "Switched to offline mode"},
        {"Time": "12:05", "Event": "Unauthorized login attempt", "Action": "Blocked"},
        {"Time": "12:10", "Event": "Sensor Alpha spoof detected", "Action": "Data ignored"},
    ]
    df_sec = pd.DataFrame(events)
    st.dataframe(df_sec)

    # Random live event generator
    if st.button("Simulate random event"):
        sample = random.choice([
            {"Event": "GPS spoofing detected", "Action": "Fallback to inertial nav"},
            {"Event": "Data packet tampering", "Action": "Discarded"},
            {"Event": "Drone Charlie under signal jamming", "Action": "Switching to autonomous mode"},
        ])
        st.warning(f"⚠️ {sample['Event']} → {sample['Action']}")

    st.markdown("✅ Future scope: real **end-to-end encryption, anti-jamming, anomaly detection**.")

# =====================================================================
st.caption("ALL RIGHTS RESERVED TO THE THIRD EYE")
