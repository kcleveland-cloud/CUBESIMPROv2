# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Physical constants (SI)
# -------------------------
MU = 3.986004418e14        # Earth's gravitational parameter, m^3/s^2
R_E = 6371e3               # Earth radius, m
OMEGA_E = 7.2921159e-5     # Earth's rotation rate, rad/s
SIGMA = 5.670374419e-8     # Stefan-Boltzmann constant, W/m^2/K^4
SOLAR_CONST = 1366         # Solar constant (approx), W/m^2
EARTH_IR = 237.0           # Approx Earth IR at LEO, W/m^2
ALBEDO = 0.3

# -------------------------
# CubeSat sim class (corrected physics)
# -------------------------
class CubeSatSim:
    def __init__(self,
                 altitude_km=400.0,
                 inclination_deg=51.6,
                 face_area_m2=0.01,
                 panel_efficiency=0.28,
                 absorptivity=0.6,
                 emissivity=0.8,
                 mass_kg=1.33,
                 drag_coeff=2.2):
        # Inputs
        self.alt_km = float(altitude_km)
        self.h = self.alt_km * 1000.0             # altitude in m
        self.inclination = np.radians(inclination_deg)
        self.face_area = float(face_area_m2)      # m^2 (single face area)
        self.panel_eff = float(panel_efficiency)  # electrical conversion eff when zenith
        self.absorptivity = float(absorptivity)
        self.emissivity = float(emissivity)
        self.mass = float(mass_kg)
        self.Cd = float(drag_coeff)

        # Derived
        self.r = R_E + self.h                      # orbital radius (m)
        self.orbital_period_s = 2.0 * np.pi * np.sqrt(self.r**3 / MU)  # s

        # view factor approximation (used for albedo/IR)
        self.view_factor = (R_E / (R_E + self.h))**2

    # -------------------------
    # Eclipse fraction (geometric)
    # -------------------------
    def eclipse_fraction(self):
        # Satellite in Earth's shadow when central angle psi satisfies cos(psi) = R_E / (R_E + h)
        # Eclipse half-angle psi = arccos(R_E / r). Fraction of orbit in shadow = psi / pi
        ratio = R_E / self.r
        # numerical safety
        ratio = np.clip(ratio, -1.0, 1.0)
        psi = np.arccos(ratio)
        frac = psi / np.pi
        # returns fraction of orbit in eclipse (0..0.5)
        return float(frac)

    # -------------------------
    # Orbital kinematics (3D)
    # -------------------------
    def simulate_orbit_3d(self, num_points=400):
        # produce coordinates in kilometers for plotting convenience
        theta = np.linspace(0.0, 2.0 * np.pi, num_points)
        x = (self.r / 1000.0) * np.cos(theta)                                  # km
        y = (self.r / 1000.0) * np.sin(theta) * np.cos(self.inclination)       # km
        z = (self.r / 1000.0) * np.sin(theta) * np.sin(self.inclination)       # km
        return x, y, z, theta

    # -------------------------
    # Ground track (account for Earth's rotation)
    # -------------------------
    def ground_track(self, num_points=400):
        # time array for one orbit in seconds
        t = np.linspace(0.0, self.orbital_period_s, num_points)
        # mean motion (rad/s)
        n = 2.0 * np.pi / self.orbital_period_s
        # satellite longitude progression relative to inertial frame:
        lon_inertial = (n * t)  # rad
        # convert to Earth-fixed longitude: subtract Earth's rotation OMEGA_E * t
        lon_earth_fixed = lon_inertial - OMEGA_E * t
        # convert to degrees and wrap to [-180,180]
        lon_deg = (np.degrees(lon_earth_fixed) + 180.0) % 360.0 - 180.0
        lat_deg = np.degrees(np.arcsin(np.sin(self.inclination) * np.sin(n * t)))
        return lon_deg, lat_deg

    # -------------------------
    # Power model (physically consistent)
    # - Geometric eclipse fraction
    # - Two modes: body-mounted (random orientation) or sun-pointing (max)
    # -------------------------
    def power_generation(self, mode='body', panel_area_m2=None):
        """
        mode: 'body' (body-mounted, averaged cosine effect)
              'sunpoint' (best-case, panel normal to Sun during sunlit portion)
        panel_area_m2: effective solar array area in m^2 (if None, uses face_area)
        Returns: avg_generated_W, eclipse_fraction
        """
        A = panel_area_m2 if panel_area_m2 is not None else self.face_area
        efrac = self.eclipse_fraction()

        if mode == 'sunpoint':
            # If sun-tracking and kept normal during sunlit fraction:
            # instantaneous P = SOLAR_CONST * A * panel_eff, averaged over orbit: multiply by (1 - eclipse_fraction)
            P_avg = SOLAR_CONST * A * self.panel_eff * (1.0 - efrac)
        else:
            # body-mounted panel: assume random orientation over spin/orbit -> average cosine factor = 0.5
            # instantaneous mean over sunlit fraction: SOLAR_CONST * A * cos_mean * panel_eff
            cos_mean = 0.5
            P_avg = SOLAR_CONST * A * cos_mean * self.panel_eff * (1.0 - efrac)

        return float(P_avg), float(efrac)

    # -------------------------
    # Thermal equilibrium temperatures (per-face approximations)
    # Using: (Q_abs) = ε σ A_rad T^4  --> T = (Q_abs / (ε σ A_rad))^(1/4)
    # Q_abs includes direct sunlight (reduced by eclipse), albedo (view_factor), and Earth IR (view_factor)
    # -------------------------
    def thermal_equilibrium(self, A_face_m2=None, A_rad_m2=None, eclipse_frac_override=None):
        A_face = A_face_m2 if A_face_m2 is not None else self.face_area
        # rough radiating area for a small cube: approximate 6 * face area if all faces radiate
        A_rad = A_rad_m2 if A_rad_m2 is not None else 6.0 * self.face_area

        if eclipse_frac_override is None:
            efrac = self.eclipse_fraction()
        else:
            efrac = float(eclipse_frac_override)

        # absorbed from direct sun on a sunlit face (when sunlit): S * α * A_face
        # average over orbit multiply by (1 - efrac)
        Q_solar_avg = SOLAR_CONST * self.absorptivity * A_face * (1.0 - efrac)
        # absorbed albedo: albedo * S * α * A_face * view_factor * (1 - efrac)
        Q_albedo = ALBEDO * SOLAR_CONST * self.absorptivity * A_face * self.view_factor * (1.0 - efrac)
        # Earth IR absorption: EARTH_IR * emissivity * A_face * view_factor
        Q_ir = EARTH_IR * self.emissivity * A_face * self.view_factor

        Q_abs_total = Q_solar_avg + Q_albedo + Q_ir

        # equilibrium T in Kelvin
        T_k = (Q_abs_total / (self.emissivity * SIGMA * A_rad)) ** 0.25
        T_c = T_k - 273.15
        return {
            'Q_solar_avg_W': float(Q_solar_avg),
            'Q_albedo_W': float(Q_albedo),
            'Q_ir_W': float(Q_ir),
            'Q_abs_total_W': float(Q_abs_total),
            'T_equilibrium_C': float(T_c)
        }

    # -------------------------
    # Basic orbital energy / period convenience
    # -------------------------
    def get_orbital_period_min(self):
        return float(self.orbital_period_s / 60.0)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Corrected CubeSat Simulator", layout="wide")
st.title("Corrected CubeSat Simulator — Physics Fixed")

# Sidebar params
with st.sidebar:
    st.header("Mission parameters")
    altitude = st.slider("Orbit altitude (km)", 200, 800, value=400)
    inclination_deg = st.slider("Inclination (deg)", 0.0, 90.0, value=51.6)
    face_area = st.number_input("Face area (m²)", min_value=0.001, max_value=1.0, value=0.01, step=0.001)
    panel_eff = st.number_input("Panel conversion efficiency (0-1)", min_value=0.01, max_value=0.6, value=0.28, step=0.01)
    mission_days = st.number_input("Mission duration (days)", min_value=1, max_value=365, value=30)
    anim_speed = st.slider("Animation speed multiplier", 0.1, 5.0, 1.0)

# instantiate sim (with corrected parameters)
sim = CubeSatSim(altitude_km=altitude,
                 inclination_deg=inclination_deg,
                 face_area_m2=face_area,
                 panel_efficiency=panel_eff,
                 absorptivity=0.6,
                 emissivity=0.8)

# Tabs
tab1, tab2, tab3 = st.tabs(["Orbit & Ground Track", "Power", "Thermal"])

# -------------------------
# Tab 1: Orbit & Ground track
# -------------------------
with tab1:
    st.subheader("Orbit geometry (corrected units & Earth rotation)")

    # orbit 3D
    x_km, y_km, z_km, theta = sim.simulate_orbit_3d(num_points=500)
    lon_deg, lat_deg = sim.ground_track(num_points=500)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**3D orbit (km)**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter3d(x=x_km, y=y_km, z=z_km, mode='lines', line=dict(color='orange', width=3)))
        fig3.update_layout(scene=dict(aspectmode='data',
                                     xaxis=dict(showticklabels=False),
                                     yaxis=dict(showticklabels=False),
                                     zaxis=dict(showticklabels=False)),
                           height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("**Ground track (one orbit)**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scattergeo(lon=lon_deg, lat=lat_deg, mode='lines', line=dict(color='royalblue')))
        fig2.update_layout(geo=dict(projection_type='natural earth'), height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # show computed orbital period and eclipse fraction
    st.markdown("### Orbital numbers")
    op_min = sim.get_orbital_period_min()
    efrac = sim.eclipse_fraction()
    st.metric("Orbital period (min)", f"{op_min:.2f}")
    st.metric("Eclipse fraction (orbit)", f"{efrac:.3f} (→ {efrac*100:.1f}%)")

# -------------------------
# Tab 2: Power (corrected)
# -------------------------
with tab2:
    st.subheader("Power Budget (physically consistent)")

    mode = st.radio("Panel attitude mode", options=['body-mounted (avg)', 'sun-pointing (best)'])
    panel_area = st.number_input("Solar array area (m²)", min_value=0.001, max_value=1.0, value=face_area, step=0.001)

    mode_key = 'body' if mode.startswith('body') else 'sunpoint'
    avg_gen_w, efrac = sim.power_generation(mode=mode_key, panel_area_m2=panel_area)

    # battery simple sim over mission days (Wh)
    battery_wh = st.number_input("Battery capacity (Wh)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
    start_soc_pct = st.slider("Start SoC (%)", 0, 100, 50)

    # convert continuous generation and consumption to daily Wh (assume constant consumption)
    # consumption baseline (payload + bus)
    consumption_w = st.number_input("Baseline bus consumption (W)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)

    seconds_per_day = 86400.0
    daily_gen_wh = avg_gen_w * seconds_per_day / 3600.0
    daily_cons_wh = consumption_w * seconds_per_day / 3600.0
    net_daily_wh = daily_gen_wh - daily_cons_wh

    st.markdown(f"**Avg generated (W)**: {avg_gen_w:.3f} W  (panel area {panel_area:.4f} m²)  — eclipse fraction {efrac:.3f}")

    # run simple SoC simulation over mission_days
    days = np.arange(1, int(mission_days) + 1)
    soc_wh = []
    soc = start_soc_pct / 100.0 * battery_wh
    for d in days:
        soc += net_daily_wh
        # clamp
        soc = min(max(soc, 0.0), battery_wh)
        soc_wh.append(100.0 * soc / battery_wh)
    df_soc = pd.DataFrame({"Day": days, "SoC (%)": soc_wh})

    # show key numbers and plots
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg gen (W)", f"{avg_gen_w:.3f}")
    col_b.metric("Daily generation (Wh/day)", f"{daily_gen_wh:.1f}")
    col_c.metric("Daily consumption (Wh/day)", f"{daily_cons_wh:.1f}")

    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True, title="Battery SoC over mission"), use_container_width=True)

    final_soc = df_soc["SoC (%)"].iloc[-1]
    if final_soc < 20.0:
        st.warning(f"Projected final SoC {final_soc:.1f}% < 20% — consider larger battery or lower consumption.")
    else:
        st.success(f"Projected final SoC {final_soc:.1f}%")

# -------------------------
# Tab 3: Thermal (corrected)
# -------------------------
with tab3:
    st.subheader("Thermal equilibrium (radiative balance)")

    A_face = st.number_input("Thermal face area for absorption (m²)", min_value=0.001, max_value=0.5, value=face_area, step=0.001)
    A_rad = st.number_input("Radiating area (m²)", min_value=0.001, max_value=3.0, value=6.0 * face_area, step=0.01)

    thermal = sim.thermal_equilibrium(A_face_m2=A_face, A_rad_m2=A_rad)
    st.write("Components of absorbed flux (W):")
    st.write(pd.DataFrame([{
        "Q_solar_avg_W": thermal['Q_solar_avg_W'],
        "Q_albedo_W": thermal['Q_albedo_W'],
        "Q_ir_W": thermal['Q_ir_W'],
        "Q_abs_total_W": thermal['Q_abs_total_W']
    }]))
    st.metric("Equilibrium temperature (°C)", f"{thermal['T_equilibrium_C']:.2f} °C")

    # small bar chart
    fig_bar = px.bar(pd.DataFrame({
        'Component': ['Solar (avg)', 'Albedo (avg)', 'Earth IR'],
        'W': [thermal['Q_solar_avg_W'], thermal['Q_albedo_W'], thermal['Q_ir_W']]
    }), x='Component', y='W', title="Absorbed power components (W)")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("Physics corrections applied: SI units internally, geometric eclipse fraction, Earth rotation in ground track, average-power model for body-mounted panels, Stefan–Boltzmann thermal balance.")
