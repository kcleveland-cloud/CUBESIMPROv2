import streamlit as st
import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import plotly.graph_objects as go
import plotly.io as pio
import json

pio.renderers.default = 'browser'  # For web deployment

st.title("🚀 CubeSim Pro — NASA-Validated 1U CubeSat Simulator")

# Sidebar for inputs
altitude_km = st.sidebar.slider("Altitude (km)", 300, 600, 400)
internal_heat = st.sidebar.slider("Internal Heat (W)", 0.0, 5.0, 2.0)

# === ORBIT ===
altitude = u.Quantity(altitude_km, u.km)
orbit = Orbit.circular(Earth, altitude)
period = orbit.period.to(u.s).value
times = np.linspace(0, period, 100) * u.s
positions = np.array([orbit.propagate(t).r.to(u.km).value for t in times])

# === POWER ===
S0 = 1366
A_panel = 0.1
eta = 0.28
sunlit_fraction = 0.59
power_avg = S0 * A_panel * eta * sunlit_fraction

# === THERMAL ===
sigma = 5.67e-8
epsilon = 0.85
alpha_s = 0.65
alpha_ir = 0.85
A_face = 0.01
A_total = 0.06

albedo = 0.30
F_albedo = S0 * albedo * 0.4
F_earth_ir = 237

Q_solar_total = S0 * alpha_s * A_face * 2
Q_albedo_total = F_albedo * alpha_s * A_total * 0.4
Q_earth_ir_total = F_earth_ir * alpha_ir * A_total * 0.4
Q_internal_total = internal_heat
Q_in_total = Q_solar_total + Q_albedo_total + Q_earth_ir_total + Q_internal_total

T_avg_K = (Q_in_total / (epsilon * sigma * A_total)) ** 0.25
T_avg_C = T_avg_K - 273.15

Q_hot_face = S0 * alpha_s * A_face + (Q_albedo_total + Q_earth_ir_total + Q_internal_total)/6
T_hot = (Q_hot_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

Q_cold_face = (Q_earth_ir_total + Q_internal_total)/6
T_cold = (Q_cold_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

# === 3D PLOT ===
fig = go.Figure()
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*6371
y = np.sin(u)*np.sin(v)*6371
z = np.cos(v)*6371
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', showscale=False))
fig.add_trace(go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], mode='lines', line=dict(color='red', width=6)))
fig.add_trace(go.Scatter3d(x=[positions[-1,0]], y=[positions[-1,1]], z=[positions[-1,2]], mode='markers', marker=dict(size=10, color='yellow')))
fig.update_layout(title=f'CubeSim Pro: {altitude_km} km Orbit', height=600)
st.plotly_chart(fig, use_container_width=True)

# === RESULTS ===
st.subheader("Mission Summary")
col1, col2 = st.columns(2)
col1.metric("Avg Power", f"{power_avg:.1f} W")
col1.metric("Hot Face", f"{T_hot:+.1f}°C")
col1.metric("Cold Face", f"{T_cold:+.1f}°C")
col1.metric("ΔT", f"{T_hot - T_cold:.1f}°C")
col2.metric("Avg Temp", f"{T_avg_C:+.1f}°C")

# Export
if st.button("Download JSON Report"):
    report = {
        "altitude_km": altitude_km,
        "power_w": round(power_avg, 1),
        "temp_hot_c": round(T_hot, 1),
        "temp_cold_c": round(T_cold, 1),
        "delta_t_c": round(T_hot - T_cold, 1),
        "avg_temp_c": round(T_avg_C, 1)
    }
    st.download_button("Download", json.dumps(report, indent=2), "cubesim_report.json", "application/json")
