# app.py
import time, math, random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import plotly.express as px

# ---------------- Page Config ----------------
st.set_page_config(page_title="A4S+ Prototype", layout="wide")

# ---------------- Custom UI ----------------
st.markdown("""
<style>
body {background-color:#0a0a0a;}
h1,h2,h3 {color:#22c55e;}
.card {
    padding: 15px;
    border-radius: 15px;
    background: #111;
    border: 1px solid #22c55e33;
    text-align: center;
}
.metric-big {font-size: 28px; font-weight: bold; color:#22c55e;}
.small {color:#aaa;}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🛡 A4S+ — Soldier Safety System")
st.markdown("### *AI + Digital Twin + UAV Intelligence*")

# ---------------- Tabs ----------------
tabs = st.tabs([
    "🧠 Health AI",
    "🗺 Safe Routing",
    "🚁 Drone Swarm",
    "🔐 Security"
])

# =====================================================================
# 1) Soldier Health Alerts
# =====================================================================
with tabs[0]:
    st.subheader("Soldier Health Monitoring")

    try:
        df = pd.read_csv("soldier_vitals.csv")
    except:
        st.error("CSV file missing. Add 'soldier_vitals.csv'")
        st.stop()

    # ---------------- Risk Logic ----------------
    def risk_score(row):
        score, reasons = 0, []

        if row["spo2"] < 60:
            score += 50; reasons.append("Critical SpO2")
        elif row["spo2"] < 70:
            score += 25; reasons.append("Low SpO2")

        if row["heart_rate"] > 115:
            score += 20; reasons.append("High HR")

        if row["skin_temp_c"] <= -12:
            score += 20; reasons.append("Hypothermia risk")

        if row["altitude_m"] >= 5000:
            score += 10; reasons.append("High altitude")

        if "prev_heart_rate" in df.columns:
            if row["heart_rate"] < row["prev_heart_rate"] - 30:
                score += 30; reasons.append("Sudden HR drop")

        return score, ", ".join(reasons)

    df[["risk_score", "reasons"]] = df.apply(
        lambda r: pd.Series(risk_score(r)), axis=1
    )

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 19, 49, 1000],
        labels=["Low", "Medium", "High"]
    )

    # ---------------- Metrics ----------------
    col1, col2, col3 = st.columns(3)

    high = len(df[df["risk_level"] == "High"])
    med = len(df[df["risk_level"] == "Medium"])
    total = len(df)

    col1.markdown(f"<div class='card'><div class='metric-big'>{high}</div>High Risk</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='metric-big'>{med}</div>Medium Risk</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='metric-big'>{total}</div>Total Soldiers</div>", unsafe_allow_html=True)

    # ---------------- Styling Fix ----------------
    def color_risk(val):
        if val == "High":
            return "background-color: #fecaca"
        elif val == "Medium":
            return "background-color: #fde68a"
        return "background-color: #bbf7d0"

    st.write(df.style.map(color_risk, subset=["risk_level"]))

    # ---------------- Alert ----------------
    high_risk = df[df["risk_level"] == "High"]
    if not high_risk.empty:
        st.error(f"🚨 HIGH RISK: {', '.join(high_risk['name'])}")
    else:
        st.success("All soldiers stable")

    # ---------------- Chart ----------------
    st.line_chart(df[["spo2", "heart_rate", "skin_temp_c"]])

# =====================================================================
# 2) Safe Route Planner
# =====================================================================
with tabs[1]:
    st.subheader("AI Route Planning")

    grid_size = st.slider("Grid Size", 10, 40, 20)
    danger = st.slider("Danger Density", 0.0, 0.5, 0.25)

    grid = np.random.rand(grid_size, grid_size)
    grid[grid < danger] = 100
    grid[grid >= danger] = 0

    start, goal = (0,0), (grid_size-1, grid_size-1)

    from heapq import heappush, heappop

    def astar(grid):
        n = len(grid)
        open_list = [(0, start)]
        visited = {}

        while open_list:
            cost, node = heappop(open_list)
            if node == goal:
                return True

            if node in visited:
                continue
            visited[node] = True

            r, c = node
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n:
                    if grid[nr][nc] < 100:
                        heappush(open_list, (cost+1, (nr,nc)))
        return False

    if astar(grid):
        st.success("Safe Path Found ✅")
    else:
        st.error("No Path ❌")

    st.plotly_chart(px.imshow(grid, color_continuous_scale="Reds"))

# =====================================================================
# 3) Drone Swarm
# =====================================================================
with tabs[2]:
    st.subheader("Drone Swarm Simulation")

    N = st.slider("Grid", 10, 30, 20)
    drones = st.slider("Drones", 2, 6, 3)

    grid = np.zeros((N, N))
    for i in range(drones):
        x, y = random.randint(0,N-1), random.randint(0,N-1)
        grid[y][x] = 1

    fig, ax = plt.subplots()
    ax.imshow(grid)
    st.pyplot(fig)

# =====================================================================
# 4) Security
# =====================================================================
with tabs[3]:
    st.subheader("Secure Communication")

    msg = st.text_input("Enter message", "Heart=90")

    if st.button("Encrypt"):
        key = Fernet.generate_key()
        f = Fernet(key)
        enc = f.encrypt(msg.encode())
        dec = f.decrypt(enc).decode()

        st.write("Encrypted:", enc)
        st.write("Decrypted:", dec)

# =====================================================================
st.caption("A4S+ Defense Prototype | Built for real-world battlefield AI")
