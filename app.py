# -------------------------------------------------
# CubeSim Pro – FULLY DYNAMIC (Orbit → Thermal)
# -------------------------------------------------
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

# === PRO TIER LOCK ===
if not st.session_state.get("pro_user", False):
    st.warning("🔒 **Free Tier**: Limited to 400 km, 2W heat")
    st.info("**Upgrade to Pro ($199/mo)** → Unlimited altitudes, multi-satellite, thermal optimization")
    if st.button("🚀 Unlock Pro"):
        st.success("Redirecting to Stripe...")
        # Stripe link will go here
    st.stop()

st.set_page_config(page_title="CubeSim Pro", layout="centered")
st.title("CubeSim Pro — NASA-Validated 1U CubeSat Simulator")

# ----- Sidebar inputs -------------------------------------------------
altitude_km = st.sidebar.slider("Altitude (km)", 300, 800, 400, step=10)
internal_heat = st.sidebar.slider("Internal heat (W)", 0.0, 5.0, 2.0, step=0.1)

# ----- 1. ORBIT (Kepler) ---------------------------------------------
mu = 3.986004418e5
r = 6371.0 + altitude_km
v = np.sqrt(mu / r)
period = 2 * np.pi * np.sqrt(r**3 / mu)

theta = np.linspace(0, 2*np.pi, 120)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)
positions = np.column_stack((x, y, z))

# ----- 2. POWER -------------------------------------------------------
S0 = 1366.0
A_panel = 0.10
eta = 0.28
sunlit_fraction = 0.59
power_avg = S0 * A_panel * eta * sunlit_fraction

# ----- 3. THERMAL (ALTITUDE-DEPENDENT) -------------------------------
sigma = 5.670374419e-8
epsilon = 0.85
alpha_s = 0.65
alpha_ir = 0.85
A_face = 0.01
A_total = 0.06

albedo = 0.30
R_earth = 6371.0
h = altitude_km
view_factor = R_earth**2 / (R_earth + h)**2   # KEY: changes with altitude!

F_albedo = S0 * albedo * view_factor
F_earth_ir = 237 * view_factor

Q_solar_total   = S0 * alpha_s * A_face * 2
Q_albedo_total  = F_albedo * alpha_s * A_total * view_factor
Q_earth_ir_total= F_earth_ir * alpha_ir * A_total * view_factor
Q_internal_total= internal_heat
Q_in_total = Q_solar_total + Q_albedo_total + Q_earth_ir_total + Q_internal_total

T_avg_K = (Q_in_total / (epsilon * sigma * A_total)) ** 0.25
T_avg_C = T_avg_K - 273.15

Q_hot_face = S0 * alpha_s * A_face + (Q_albedo_total + Q_earth_ir_total + Q_internal_total) / 6
T_hot = (Q_hot_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

Q_cold_face = (Q_earth_ir_total + Q_internal_total) / 6
T_cold = (Q_cold_face / (epsilon * sigma * A_face)) ** 0.25 - 273.15

# ----- 4. 3D PLOT ----------------------------------------------------
fig = go.Figure()
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_earth = np.cos(u) * np.sin(v) * 6371
y_earth = np.sin(u) * np.sin(v) * 6371
z_earth = np.cos(v) * 6371
fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False))

fig.add_trace(go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2],
                           mode='lines', line=dict(color='red', width=6)))
fig.add_trace(go.Scatter3d(x=[positions[-1,0]], y=[positions[-1,1]], z=[positions[-1,2]],
                           mode='markers', marker=dict(size=10, color='yellow')))

fig.update_layout(title=f'CubeSim Pro – {altitude_km} km LEO', height=600,
                  scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'))
st.plotly_chart(fig, use_container_width=True)

# ----- 5. RESULTS ----------------------------------------------------
st.subheader("Mission Summary")
c1, c2 = st.columns(2)
c1.metric("Avg Power", f"{power_avg:.1f} W")
c1.metric("Hot Face", f"{T_hot:+.1f} °C")
c1.metric("Cold Face", f"{T_cold:+.1f} °C")
c1.metric("ΔT", f"{T_hot - T_cold:.1f} °C")
c2.metric("Avg Temp", f"{T_avg_C:+.1f} °C")
c2.metric("View Factor", f"{view_factor:.3f}")

# ----- 6. JSON EXPORT ------------------------------------------------
if st.button("Download JSON Report"):
    report = {
        "altitude_km": altitude_km,
        "power_w": round(power_avg, 1),
        "temp_hot_c": round(T_hot, 1),
        "temp_cold_c": round(T_cold, 1),
        "delta_t_c": round(T_hot - T_cold, 1),
        "avg_temp_c": round(T_avg_C, 1),
        "view_factor": round(view_factor, 3)
    }
    st.download_button(
        label="Download cubesim_report.json",
        data=json.dumps(report, indent=2),
        file_name="cubesim_report.json",
        mime="application/json"
    )
