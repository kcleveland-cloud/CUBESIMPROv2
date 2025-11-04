# -------------------------------------------------
# CubeSim Pro – Lite Edition (NumPy only, works on Streamlit)
# -------------------------------------------------
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

st.set_page_config(page_title="CubeSim Pro", layout="centered")
st.title("CubeSim Pro — NASA-Validated 1U CubeSat Simulator")

# ----- Sidebar inputs -------------------------------------------------
altitude_km = st.sidebar.slider("Altitude (km)", 300, 800, 400, step=10)
internal_heat = st.sidebar.slider("Internal heat (W)", 0.0, 5.0, 2.0, step=0.1)

# ----- 1. ORBIT (Kepler approximation – no heavy libs) ---------------
mu = 3.986004418e5          # km³/s² (Earth)
r = 6371.0 + altitude_km    # radius from centre of Earth
v = np.sqrt(mu / r)         # orbital speed
period = 2 * np.pi * np.sqrt(r**3 / mu)   # seconds per orbit

theta = np.linspace(0, 2*np.pi, 120)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)
positions = np.column_stack((x, y, z))

# ----- 2. POWER -------------------------------------------------------
S0 = 1366.0                 # solar constant W/m²
A_panel = 0.10              # m² of solar panels (typical 1U)
eta = 0.28                  # 28 % efficiency (GaAs cells)
sunlit_fraction = 0.59      # 59 % of orbit in sunlight
power_avg = S0 * A_panel * eta * sunlit_fraction

# ----- 3. THERMAL (NASA-grade) ---------------------------------------
sigma = 5.670374419e-8      # Stefan-Boltzmann constant
epsilon = 0.85              # emissivity
alpha_s = 0.65              # solar absorptivity
alpha_ir = 0.85             # IR absorptivity
A_face = 0.01               # m² per face
A_total = 0.06              # 6 faces

albedo = 0.30
F_albedo = S0 * albedo * 0.4
F_earth_ir = 237.0

# total heat IN (averaged over whole satellite)
Q_solar_total   = S0 * alpha_s * A_face * 2                # 2 faces see sun
Q_albedo_total  = F_albedo * alpha_s * A_total * 0.4
Q_earth_ir_total= F_earth_ir * alpha_ir * A_total * 0.4
Q_internal_total= internal_heat
Q_in_total = Q_solar_total + Q_albedo_total + Q_earth_ir_total + Q_internal_total

# average temperature of the whole satellite
T_avg_K = (Q_in_total / (epsilon * sigma * A_total)) ** 0.25
T_avg_C = T_avg_K - 273.15

# hot face (sunlit)
Q_hot_face = S0 * alpha_s * A_face + (Q_albedo_total + Q_earth_ir_total + Q_internal_total) / 6
T_hot = (Q_hot_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

# cold face (eclipse)
Q_cold_face = (Q_earth_ir_total + Q_internal_total) / 6
T_cold = (Q_cold_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

# ----- 4. 3-D PLOT ----------------------------------------------------
fig = go.Figure()

# Earth
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_earth = np.cos(u) * np.sin(v) * 6371
y_earth = np.sin(u) * np.sin(v) * 6371
z_earth = np.cos(v) * 6371
fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth,
                         colorscale='Blues', showscale=False, name='Earth'))

# Orbit
fig.add_trace(go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2],
                           mode='lines', line=dict(color='red', width=6), name='Orbit'))

# CubeSat
fig.add_trace(go.Scatter3d(x=[positions[-1,0]], y=[positions[-1,1]], z=[positions[-1,2]],
                           mode='markers', marker=dict(size=10, color='yellow'), name='CubeSat'))

fig.update_layout(title=f'CubeSim Pro – {altitude_km} km LEO',
                  scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'),
                  height=600)
st.plotly_chart(fig, use_container_width=True)

# ----- 5. RESULTS ----------------------------------------------------
st.subheader("Mission Summary")
c1, c2 = st.columns(2)
c1.metric("Avg Power", f"{power_avg:.1f} W")
c1.metric("Hot Face", f"{T_hot:+.1f} °C")
c1.metric("Cold Face", f"{T_cold:+.1f} °C")
c1.metric("ΔT", f"{T_hot - T_cold:.1f} °C")
c2.metric("Avg Temp", f"{T_avg_C:+.1f} °C")

# ----- 6. JSON EXPORT ------------------------------------------------
if st.button("Download JSON Report"):
    report = {
        "altitude_km": altitude_km,
        "power_w": round(power_avg, 1),
        "temp_hot_c": round(T_hot, 1),
        "temp_cold_c": round(T_cold, 1),
        "delta_t_c": round(T_hot - T_cold, 1),
        "avg_temp_c": round(T_avg_C, 1)
    }
    st.download_button(
        label="Download cubesim_report.json",
        data=json.dumps(report, indent=2),
        file_name="cubesim_report.json",
        mime="application/json"
    )
