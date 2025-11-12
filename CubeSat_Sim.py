# app.py - GeneSat-1–tuned CubeSat Simulator with Play buttons restored
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Constants (SI)
# -------------------------
MU = 3.986004418e14       # Earth's gravitational parameter (m^3 / s^2)
R_E = 6371e3              # Earth radius (m)
SIGMA = 5.670374419e-8    # Stefan-Boltzmann (W / m^2 / K^4)
SOLAR_CONST = 1366.0      # W/m^2 (approx)
EARTH_IR = 237.0          # W/m^2 (approx)
ALBEDO = 0.3
OMEGA_E = 7.2921159e-5    # Earth's rotation rate (rad/s)

# -------------------------
# GeneSat-1 baseline parameters (from public mission docs)
# These defaults aim to represent GeneSat-1-like behavior (user may change)
# -------------------------
GENESAT_DEFAULTS = {
    "altitude_km": 460.0,        # typical reported altitude band
    "inclination_deg": 40.0,     # ~40°
    "mass_kg": 4.6,              # reported approximate launch mass
    "face_area_m2": 0.03,        # per-face area approx (1U face ~0.01 -> 3U ~0.03)
    "panel_efficiency": 0.25,    # practical cell efficiency for body mounted arrays
    "absorptivity": 0.65,
    "emissivity": 0.83
}

# -------------------------
# CubeSatSim class (GeneSat-tuned physics)
# -------------------------
class CubeSatSim:
    def __init__(self,
                 altitude_km=GENESAT_DEFAULTS["altitude_km"],
                 inclination_deg=GENESAT_DEFAULTS["inclination_deg"],
                 mass_kg=GENESAT_DEFAULTS["mass_kg"],
                 face_area_m2=GENESAT_DEFAULTS["face_area_m2"],
                 panel_efficiency=GENESAT_DEFAULTS["panel_efficiency"],
                 absorptivity=GENESAT_DEFAULTS["absorptivity"],
                 emissivity=GENESAT_DEFAULTS["emissivity"]):
        # store inputs
        self.h = float(altitude_km) * 1000.0            # altitude (m)
        self.alt_km = float(altitude_km)
        self.inclination = np.radians(float(inclination_deg))
        self.mass = float(mass_kg)
        self.face_area = float(face_area_m2)
        self.panel_eff = float(panel_efficiency)
        self.absorptivity = float(absorptivity)
        self.emissivity = float(emissivity)

        # derived
        self.r = R_E + self.h                            # orbital radius (m)
        self.orbital_period_s = 2.0 * np.pi * np.sqrt(self.r**3 / MU)
        # view factor (solid-angle fraction approximation) -- geometric
        # Use (1 - cos(theta))/2 where theta = arccos(R_E / r) approximates Earth subtended cap fraction.
        ratio = np.clip(R_E / self.r, -1.0, 1.0)
        psi = np.arccos(ratio)         # half-angle in radians for Earth's limb as seen by sat
        # view_factor_earth approximates fraction of sky hemisphere occupied by Earth (0..1)
        self.view_factor_earth = (1.0 - np.cos(psi)) / 2.0

    # geometric eclipse fraction: half-angle psi; fraction in shadow = psi / pi
    def eclipse_fraction(self):
        ratio = np.clip(R_E / self.r, -1.0, 1.0)
        psi = np.arccos(ratio)
        frac = float(psi / np.pi)   # 0..0.5
        return frac

    # 3D orbit points (km) and theta for phase
    def simulate_orbit_3d(self, num_points=400):
        theta = np.linspace(0.0, 2.0 * np.pi, num_points)
        r_km = self.r / 1000.0
        x = r_km * np.cos(theta)
        y = r_km * np.sin(theta) * np.cos(self.inclination)
        z = r_km * np.sin(theta) * np.sin(self.inclination)
        return x, y, z, theta

    # ground track (account for Earth's rotation)
    def ground_track(self, num_points=400):
        t = np.linspace(0.0, self.orbital_period_s, num_points)
        n = 2.0 * np.pi / self.orbital_period_s
        lon_inertial = n * t
        lon_earth_fixed = lon_inertial - OMEGA_E * t
        lon_deg = (np.degrees(lon_earth_fixed) + 180.0) % 360.0 - 180.0
        lat_deg = np.degrees(np.arcsin(np.sin(self.inclination) * np.sin(n * t)))
        return lon_deg, lat_deg

    # Power generation model (body-mounted vs sun-pointing)
    def power_generation(self, mode='body-mounted', panel_area_m2=None):
        A = panel_area_m2 if panel_area_m2 is not None else self.face_area
        efrac = self.eclipse_fraction()

        if mode == 'sun-pointing':
            # idealized: panel normal to sun during sunlit fraction
            p_avg = SOLAR_CONST * A * self.panel_eff * (1.0 - efrac)
        else:
            # body-mounted (no pointing): average cosine factor for a randomly oriented flat plate ~0.5
            cos_mean = 0.5
            p_avg = SOLAR_CONST * A * cos_mean * self.panel_eff * (1.0 - efrac)

        return float(p_avg), float(efrac)

    # Thermal balance per-face (averaged over orbit)
    def thermal_balance(self, A_face_m2=None, A_rad_m2=None):
        A_face = A_face_m2 if A_face_m2 is not None else self.face_area
        A_rad = A_rad_m2 if A_rad_m2 is not None else 6.0 * self.face_area

        efrac = self.eclipse_fraction()
        # solar absorbed (averaged over orbit)
        Q_solar_avg = SOLAR_CONST * self.absorptivity * A_face * (1.0 - efrac)
        # albedo contribution
        Q_albedo = ALBEDO * SOLAR_CONST * self.absorptivity * A_face * self.view_factor_earth * (1.0 - efrac)
        # Earth IR
        Q_ir = EARTH_IR * self.emissivity * A_face * self.view_factor_earth

        Q_total = Q_solar_avg + Q_albedo + Q_ir
        # equilibrium temperature (Kelvin)
        T_k = (Q_total / (self.emissivity * SIGMA * A_rad)) ** 0.25
        T_c = float(T_k - 273.15)

        return {
            "Q_solar_avg_W": float(Q_solar_avg),
            "Q_albedo_W": float(Q_albedo),
            "Q_ir_W": float(Q_ir),
            "Q_abs_total_W": float(Q_total),
            "T_equilibrium_C": T_c
        }

    def orbital_period_min(self):
        return float(self.orbital_period_s / 60.0)


# -------------------------
# Streamlit app UI
# -------------------------
st.set_page_config(page_title="GeneSat-1 Tuned CubeSat Simulator", layout="wide")
st.title("GeneSat-1 Tuned CubeSat Simulator")
st.markdown("This app restores Plotly Play buttons and uses GeneSat-1–informed defaults (docs cited below).")

# --- sidebar controls ---
with st.sidebar:
    st.header("Mission / CubeSat parameters (defaults ≈ GeneSat-1)")
    altitude_km = st.number_input("Orbit altitude (km)", min_value=200.0, max_value=600.0, value=GENESAT_DEFAULTS["altitude_km"], step=1.0)
    inclination_deg = st.number_input("Inclination (°)", min_value=0.0, max_value=98.0, value=GENESAT_DEFAULTS["inclination_deg"], step=0.1)
    mass_kg = st.number_input("Mass (kg)", min_value=0.1, max_value=50.0, value=GENESAT_DEFAULTS["mass_kg"], step=0.1)
    face_area = st.number_input("Face area (m²)", min_value=0.001, max_value=0.2, value=GENESAT_DEFAULTS["face_area_m2"], step=0.001)
    panel_eff = st.slider("Panel efficiency (η)", min_value=0.05, max_value=0.35, value=GENESAT_DEFAULTS["panel_efficiency"], step=0.01)
    absorp = st.slider("Absorptivity α", min_value=0.1, max_value=1.0, value=GENESAT_DEFAULTS["absorptivity"], step=0.01)
    emis = st.slider("Emissivity ε", min_value=0.1, max_value=1.0, value=GENESAT_DEFAULTS["emissivity"], step=0.01)
    panel_mode = st.radio("Panel attitude mode", options=["body-mounted (avg)", "sun-pointing (best)"])
    mission_days = st.number_input("Mission duration (days)", min_value=1, max_value=365, value=30)
    anim_speed = st.slider("Animation speed (1 = baseline)", 0.1, 5.0, 1.0, step=0.1)

# instantiate simulator
sim = CubeSatSim(altitude_km=altitude_km,
                 inclination_deg=inclination_deg,
                 mass_kg=mass_kg,
                 face_area_m2=face_area,
                 panel_efficiency=panel_eff,
                 absorptivity=absorp,
                 emissivity=emis)

# --- tabs ---
tab_orbit, tab_power, tab_thermal = st.tabs(["3D + Ground Track", "Power", "Thermal"])

# -------------- TAB: Orbit & animations --------------
with tab_orbit:
    st.subheader("Orbit geometry and animations")

    # compute orbit and ground track
    x_km, y_km, z_km, theta = sim.simulate_orbit_3d(num_points=360)
    lon_deg, lat_deg = sim.ground_track(num_points=360)

    # provide checkboxes to conditionally show Play UI in figures
    show_play_ui = st.checkbox("Show Plotly Play buttons on figures (recommended)", value=True)

    # --- 3D wireframe + Earth surface (with frames) ---
    fig_3d = go.Figure()
    # 3D orbit path
    fig_3d.add_trace(go.Scatter3d(x=x_km, y=y_km, z=z_km, mode="lines",
                                 line=dict(color="gold", width=3), name="orbit"))
    # initial satellite marker
    fig_3d.add_trace(go.Scatter3d(x=[x_km[0]], y=[y_km[0]], z=[z_km[0]],
                                 mode="markers", marker=dict(size=6, color="red"), name="sat"))

    # optional translucent Earth surface
    # create a smooth sphere grid (km)
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    x_earth = (R_E/1000.0) * np.cos(uu) * np.sin(vv)
    y_earth = (R_E/1000.0) * np.sin(uu) * np.sin(vv)
    z_earth = (R_E/1000.0) * np.cos(vv)
    fig_3d.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, opacity=0.35, showscale=False, name="Earth"))

    # frames for the 3D orbit animation
    frames_3d = []
    step = 4
    for i in range(0, len(x_km), step):
        frames_3d.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(x=x_km[:i], y=y_km[:i], z=z_km[:i], mode="lines", line=dict(color="gold", width=3)),
                go.Scatter3d(x=[x_km[i % len(x_km)]], y=[y_km[i % len(y_km)]], z=[z_km[i % len(z_km)]],
                             mode="markers", marker=dict(size=6, color="red"))
            ]
        ))

    fig_3d.frames = frames_3d
    fig_3d.update_layout(scene=dict(aspectmode="data"),
                         margin=dict(l=0,r=0,t=0,b=0), height=600, showlegend=False)

    if show_play_ui:
        # add Plotly's built-in Play button (visible on figure)
        fig_3d.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.08,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True),
                                     fromcurrent=True, mode="immediate")]
                )]
            )]
        )

    st.plotly_chart(fig_3d, use_container_width=True)

    # --- Flat ground track with frames ---
    fig_flat = go.Figure()
    fig_flat.add_trace(go.Scattergeo(lon=lon_deg, lat=lat_deg, mode="lines", line=dict(color="royalblue", width=2)))
    fig_flat.add_trace(go.Scattergeo(lon=[lon_deg[0]], lat=[lat_deg[0]], mode="markers",
                                     marker=dict(size=6, color="red")))
    frames_flat = []
    for i in range(0, len(lon_deg), step):
        frames_flat.append(go.Frame(
            name=str(i),
            data=[
                go.Scattergeo(lon=lon_deg[:i], lat=lat_deg[:i], mode="lines", line=dict(color="yellow", width=2)),
                go.Scattergeo(lon=[lon_deg[i % len(lon_deg)]], lat=[lat_deg[i % len(lat_deg)]],
                              mode="markers", marker=dict(size=6, color="red"))
            ]
        ))
    fig_flat.frames = frames_flat
    fig_flat.update_layout(geo=dict(projection_type="equirectangular", showland=True), height=350, margin=dict(t=0))
    if show_play_ui:
        fig_flat.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.12,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True),
                                     fromcurrent=True, mode="immediate")]
                )]
            )]
        )
    st.plotly_chart(fig_flat, use_container_width=True)

    # --- Globe (orthographic) ground track ---
    fig_globe = go.Figure()
    fig_globe.add_trace(go.Scattergeo(lon=lon_deg, lat=lat_deg, mode="lines", line=dict(color="crimson", width=2)))
    fig_globe.add_trace(go.Scattergeo(lon=[lon_deg[0]], lat=[lat_deg[0]], mode="markers",
                                      marker=dict(size=6, color="red")))
    frames_globe = []
    for i in range(0, len(lon_deg), step):
        frames_globe.append(go.Frame(
            name=str(i),
            data=[
                go.Scattergeo(lon=lon_deg[:i], lat=lat_deg[:i], mode="lines", line=dict(color="yellow", width=2)),
                go.Scattergeo(lon=[lon_deg[i % len(lon_deg)]], lat=[lat_deg[i % len(lat_deg)]],
                              mode="markers", marker=dict(size=6, color="red"))
            ]
        ))
    fig_globe.frames = frames_globe
    fig_globe.update_layout(geo=dict(projection_type="orthographic", showland=True), height=350, margin=dict(t=0))
    if show_play_ui:
        fig_globe.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.14,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True),
                                     fromcurrent=True, mode="immediate")]
                )]
            )]
        )
    st.plotly_chart(fig_globe, use_container_width=True)

    # show orbital numbers
    st.markdown("### Orbital numbers")
    st.metric("Orbital period (min)", f"{sim.orbital_period_min():.2f}")
    st.metric("Eclipse fraction (orbit)", f"{sim.eclipse_fraction():.3f}")

# -------------- TAB: Power --------------
with tab_power:
    st.subheader("Power budget (GeneSat-tuned)")
    mode_key = 'sun-pointing' if panel_mode.startswith('sun') else 'body-mounted'
    panel_area = st.number_input("Solar array effective area (m²)", min_value=0.001, max_value=0.5, value=face_area, step=0.001)

    avg_gen_W, efrac = sim.power_generation(mode=('sun-pointing' if mode_key == 'sun-pointing' else 'body-mounted'),
                                            panel_area_m2=panel_area)

    # allow user to set consumption
    consumption_W = st.number_input("Average consumption (W)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    net_W = avg_gen_W - consumption_W
    st.metric("Avg generation (W)", f"{avg_gen_W:.3f}")
    st.metric("Avg consumption (W)", f"{consumption_W:.3f}")
    st.metric("Net (W)", f"{net_W:+.3f}")

    st.markdown(f"**Eclipse fraction:** {efrac:.3f} (fraction of orbit in Earth's shadow)")

    # battery simulation (simple daily energy)
    battery_Wh = st.number_input("Battery capacity (Wh)", min_value=1.0, max_value=200.0, value=30.0)
    start_soc_pct = st.slider("Start SoC (%)", 0, 100, 50)

    # daily energy conversions
    seconds_per_day = 86400.0
    daily_gen_Wh = avg_gen_W * seconds_per_day / 3600.0
    daily_cons_Wh = consumption_W * seconds_per_day / 3600.0
    net_daily_wh = daily_gen_Wh - daily_cons_Wh

    days = np.arange(1, int(mission_days) + 1)
    soc_list = []
    soc_wh = start_soc_pct / 100.0 * battery_Wh
    for d in days:
        soc_wh += net_daily_wh
        soc_wh = min(max(soc_wh, 0.0), battery_Wh)
        soc_list.append(100.0 * soc_wh / battery_Wh)
    df_soc = pd.DataFrame({"Day": days, "SoC (%)": soc_list})

    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True, title="Battery SoC over mission"), use_container_width=True)

    if df_soc["SoC (%)"].iloc[-1] < 20.0:
        st.warning("Projected final SoC < 20% — consider larger battery, pointing, or reduce power draw.")
    else:
        st.success("Battery SoC projection looks acceptable.")

# -------------- TAB: Thermal --------------
with tab_thermal:
    st.subheader("Thermal balance (radiative equilibrium)")

    A_face = st.number_input("Absorbing face area (m²)", min_value=0.001, max_value=0.5, value=face_area, step=0.001)
    A_rad = st.number_input("Radiating area (m²)", min_value=0.001, max_value=3.0, value=6.0*face_area, step=0.01)

    thermo = sim.thermal_balance(A_face_m2=A_face, A_rad_m2=A_rad)
    st.write(pd.DataFrame([{
        "Q_solar_avg_W": thermo["Q_solar_avg_W"],
        "Q_albedo_W": thermo["Q_albedo_W"],
        "Q_ir_W": thermo["Q_ir_W"],
        "Q_abs_total_W": thermo["Q_abs_total_W"]
    }]))
    st.metric("Equilibrium temperature (°C)", f"{thermo['T_equilibrium_C']:.2f}")

st.markdown("---")
st.caption("GeneSat-1 tuned defaults and orbital facts referenced from mission docs / community sources.")
