# app.py — Phase 1 CubeSat Sim with Pro-gated Save/Load/Export
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Physical constants (SI)
# =========================
MU = 3.986004418e14       # Earth's gravitational parameter (m^3/s^2)
R_E = 6371e3              # Earth radius (m)
OMEGA_E = 7.2921159e-5    # Earth rotation rate (rad/s)
SIGMA = 5.670374419e-8    # Stefan–Boltzmann (W/m^2/K^4)
SOLAR_CONST = 1366.0      # W/m^2
EARTH_IR = 237.0          # W/m^2
ALBEDO = 0.3

# =========================
# Helpers
# =========================
def clamp_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def earth_view_factor(alt_m):
    r = R_E + alt_m
    ratio = np.clip(R_E / r, -1.0, 1.0)
    psi = np.arccos(ratio)
    return (1.0 - np.cos(psi)) / 2.0

def rho_msis_simple(h_m):
    # Simple exponential atmosphere ~200–600 km band
    H = 60e3
    rho_200 = 2.5e-11
    h = np.maximum(h_m, 200e3)
    return np.clip(rho_200 * np.exp(-(h - 200e3)/H), 1e-13, None)

def eci_to_ecef_xyz(x_eci, y_eci, z_eci, t):
    """Rotate ECI about +Z by -OMEGA_E*t to get ECEF; arrays ok."""
    cosw = np.cos(OMEGA_E * t)
    sinw = np.sin(OMEGA_E * t)
    x_ecef =  cosw * x_eci + sinw * y_eci
    y_ecef = -sinw * x_eci + cosw * y_eci
    z_ecef =  z_eci
    return x_ecef, y_ecef, z_ecef

def eclipse_mask_from_eci(x_km, y_km, z_km):
    """Sun along +X; eclipse when x<0 and sqrt(y^2+z^2) < R_E."""
    r_perp = np.sqrt(y_km**2 + z_km**2)
    return (x_km < 0) & (r_perp < (R_E/1000.0))

# =========================
# GeneSat-1 preset (validation)
# =========================
GENESAT_DEFAULTS = dict(
    altitude_km=460.0,
    incl_deg=40.0,
    mass_kg=4.6,
    Cd=2.2,
    panel_area_m2=0.03,
    panel_eff=0.25,
    absorptivity=0.65,
    emissivity=0.83,
    target_avg_power_W=4.5
)

# =========================
# Core simulator
# =========================
class CubeSatSim:
    def __init__(self, altitude_km, incl_deg, mass_kg, Cd, panel_area_m2, panel_eff, absorptivity, emissivity):
        self.alt_km = float(altitude_km)
        self.h = self.alt_km * 1000.0
        self.i = np.radians(float(incl_deg))
        self.m = float(mass_kg)
        self.Cd = float(Cd)
        self.A_panel = float(panel_area_m2)
        self.eta = float(panel_eff)
        self.alpha = float(absorptivity)
        self.eps = float(emissivity)

        self.r = R_E + self.h
        self.T_orbit = 2*np.pi*np.sqrt(self.r**3/MU)   # s
        self.n = 2*np.pi/self.T_orbit                  # rad/s
        self.v = np.sqrt(MU/self.r)                    # m/s
        self.VF = earth_view_factor(self.h)

    def eclipse_fraction(self):
        ratio = np.clip(R_E / (R_E + self.h), -1.0, 1.0)
        psi = np.arccos(ratio)
        return float(psi / np.pi)  # 0..0.5

    # ---------- Orbit in ECI and synchronized timebase ----------
    def orbit_eci(self, N=720):
        """Return time array (s) for one orbit and ECI position vectors in km."""
        t = np.linspace(0.0, self.T_orbit, N, endpoint=False)
        u = self.n * t  # argument of latitude (circular orbit)
        r_km = self.r / 1000.0
        x = r_km * np.cos(u)
        y = r_km * np.sin(u) * np.cos(self.i)
        z = r_km * np.sin(u) * np.sin(self.i)
        return t, u, x, y, z

    def ground_track_from_eci(self, t, x_eci_km, y_eci_km, z_eci_km):
        """Exact ground track from SAME timebase using ECI->ECEF rotation."""
        x_m = x_eci_km * 1000.0
        y_m = y_eci_km * 1000.0
        z_m = z_eci_km * 1000.0
        x_ecef, y_ecef, z_ecef = eci_to_ecef_xyz(x_m, y_m, z_m, t)
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))
        lat = np.degrees(np.arcsin(z_ecef / np.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)))
        lon = clamp_angle_deg(lon)
        return lon, lat

    # ---------- Attitude & power ----------
    def instantaneous_power(self, attitude, t, x_km, y_km, z_km, A_panel, eta):
        # Sun dir = +X (ECI). Panel normal:
        if attitude == "body-spin":
            cos_inc = np.full_like(t, 0.5)
        elif attitude == "sun-tracking":
            nx = np.ones_like(t); ny = np.zeros_like(t); nz = np.zeros_like(t)
            cos_inc = nx  # dot with +X
        elif attitude == "nadir-pointing":
            rv = np.vstack((x_km, y_km, z_km)).T
            rnorm = np.linalg.norm(rv, axis=1, keepdims=True)
            nhat = -(rv / np.maximum(rnorm, 1e-12))
            cos_inc = nhat[:, 0]  # dot with +X
        else:
            cos_inc = np.full_like(t, 0.5)
        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        ecl = eclipse_mask_from_eci(x_km, y_km, z_km)
        cos_inc = cos_inc * (~ecl)
        P = SOLAR_CONST * A_panel * eta * cos_inc
        return P, cos_inc, ecl

    def avg_power(self, attitude, A_panel, eta):
        t, u, x, y, z = self.orbit_eci(N=1440)
        P, _, _ = self.instantaneous_power(attitude, t, x, y, z, A_panel, eta)
        return float(P.mean())

    # ---------- Thermal ----------
    def thermal_equilibrium(self, A_abs=None, A_rad=None):
        if A_abs is None:
            A_abs = self.A_panel
        if A_rad is None:
            A_rad = 6.0*self.A_panel
        efrac = self.eclipse_fraction()
        Q_solar = SOLAR_CONST * self.alpha * A_abs * (1.0 - efrac)
        Q_albedo = ALBEDO * SOLAR_CONST * self.alpha * A_abs * self.VF * (1.0 - efrac)
        Q_ir = EARTH_IR * self.eps * A_abs * self.VF
        Q_abs = Q_solar + Q_albedo + Q_ir
        T_k = (Q_abs/(self.eps*SIGMA*A_rad))**0.25
        return float(T_k - 273.15), Q_solar, Q_albedo, Q_ir, Q_abs

    # ---------- Drag decay ----------
    def drag_decay_days(self, days, A_drag=None):
        A = self.A_panel if A_drag is None else float(A_drag)
        a = R_E + self.h
        out = []
        for _ in range(int(days)):
            rho = rho_msis_simple(a - R_E)
            da_dt = - (self.Cd * A / self.m) * np.sqrt(MU * a) * rho
            a = a + da_dt * 86400.0
            if a < R_E + 120e3:
                a = R_E + 120e3
            out.append(a - R_E)
        return np.array(out) / 1000.0  # km

# =========================
# App header & subscription stub
# =========================
st.set_page_config(page_title="CubeSat Simulator — Phase 1", layout="wide")
st.title("🛰️ CubeSat Simulator — Phase 1 (GeneSat-validated, mission-tunable)")

# Session defaults
if "user" not in st.session_state: st.session_state.user = None
if "plan" not in st.session_state: st.session_state.plan = "Free"

with st.sidebar:
    st.header("Sign in (stub)")
    if st.session_state.user is None:
        email = st.text_input("Email")
        if st.button("Sign in"):
            st.session_state.user = email or "guest"
    else:
        st.success(f"Signed in as {st.session_state.user}")
        if st.button("Sign out"):
            st.session_state.user = None
            st.session_state.plan = "Free"

    st.header("Plan")
    st.markdown(f"**Current plan:** {st.session_state.plan}")
    if st.session_state.plan == "Free":
        if st.button("Upgrade to Pro ($9/mo)"):
            # Phase-1 stub: flip to Pro. Wire to Stripe in Phase-2.
            st.session_state.plan = "Pro"
    else:
        st.success("Pro features unlocked ✓")

    st.header("Preset & Validation")
    use_genesat = st.checkbox("Load GeneSat-1 defaults", True)
    auto_cal = st.checkbox("Calibrate avg power to GeneSat target (~4.5 W)", True)

    st.header("Mission Parameters")
    if use_genesat:
        altitude_km = st.slider("Altitude (km)", 200.0, 700.0, GENESAT_DEFAULTS["altitude_km"])
        incl_deg    = st.slider("Inclination (deg)", 0.0, 98.0, GENESAT_DEFAULTS["incl_deg"])
        mass_kg     = st.number_input("Mass (kg)", 0.1, 50.0, GENESAT_DEFAULTS["mass_kg"], 0.1)
        Cd          = st.slider("Drag coefficient Cd", 1.5, 3.0, GENESAT_DEFAULTS["Cd"], 0.1)
        panel_area  = st.number_input("Panel area / face (m²)", 0.001, 0.5, GENESAT_DEFAULTS["panel_area_m2"], 0.001)
        panel_eff   = st.slider("Panel efficiency η", 0.05, 0.38, GENESAT_DEFAULTS["panel_eff"], 0.01)
        absorp      = st.slider("Absorptivity α", 0.1, 1.0, GENESAT_DEFAULTS["absorptivity"], 0.01)
        emiss       = st.slider("Emissivity ε", 0.1, 1.0, GENESAT_DEFAULTS["emissivity"], 0.01)
        target_avgW = GENESAT_DEFAULTS["target_avg_power_W"]
    else:
        altitude_km = st.slider("Altitude (km)", 200.0, 2000.0, 500.0)
        incl_deg    = st.slider("Inclination (deg)", 0.0, 98.0, 51.6)
        mass_kg     = st.number_input("Mass (kg)", 0.1, 200.0, 4.0, 0.1)
        Cd          = st.slider("Drag coefficient Cd", 1.0, 3.5, 2.2, 0.1)
        panel_area  = st.number_input("Panel area / face (m²)", 0.001, 2.0, 0.05, 0.001)
        panel_eff   = st.slider("Panel efficiency η", 0.05, 0.38, 0.28, 0.01)
        absorp      = st.slider("Absorptivity α", 0.1, 1.0, 0.6, 0.01)
        emiss       = st.slider("Emissivity ε", 0.1, 1.0, 0.8, 0.01)
        target_avgW = st.number_input("Target avg power (W) for calibration", 0.1, 50.0, 4.5, 0.1)

    st.header("Attitude & Ops")
    attitude = st.radio("Attitude", ["body-spin", "sun-tracking", "nadir-pointing"])
    show_play = st.checkbox("Show Play buttons on plots", True)
    anim_speed = st.slider("Animation speed (Plotly)", 0.1, 5.0, 1.0, 0.1)
    mission_days = st.slider("Mission duration (days)", 1, 365, 60)

# Build simulator & optional power calibration
sim = CubeSatSim(
    altitude_km=altitude_km, incl_deg=incl_deg, mass_kg=mass_kg,
    Cd=Cd, panel_area_m2=panel_area, panel_eff=panel_eff,
    absorptivity=absorp, emissivity=emiss
)
cal_factor = 1.0
if auto_cal:
    P_now = sim.avg_power(attitude, sim.A_panel, sim.eta)
    if P_now > 1e-9:
        cal_factor = target_avgW / P_now

# =========================
# Tabs
# =========================
tab_orbit, tab_power, tab_thermal, tab_drag, tab_io = st.tabs([
    "3D + Ground Track (aligned)", "Power", "Thermal", "Drag", "Save/Load & Export"
])

# =========================
# TAB 1: ORBIT + ALIGNED GROUND TRACK
# =========================
with tab_orbit:
    st.subheader("3D Orbit (ECI) + Aligned Ground Track (ECEF)")
    t, u, x_km, y_km, z_km = sim.orbit_eci(N=720)
    lon_deg, lat_deg = sim.ground_track_from_eci(t, x_km, y_km, z_km)
    eclipsed = eclipse_mask_from_eci(x_km, y_km, z_km)

    # Sunlit/Eclipse segmented lines (static)
    x_sun, y_sun, z_sun = x_km.copy(), y_km.copy(), z_km.copy()
    x_sun[eclipsed] = None; y_sun[eclipsed] = None; z_sun[eclipsed] = None
    x_ecl, y_ecl, z_ecl = x_km.copy(), y_km.copy(), z_km.copy()
    x_ecl[~eclipsed] = None; y_ecl[~eclipsed] = None; z_ecl[~eclipsed] = None

    # 3D figure (Earth persists)
    fig3d = go.Figure()
    # trace 0: Earth surface
    u_s = np.linspace(0, 2*np.pi, 60)
    v_s = np.linspace(0, np.pi, 30)
    UU, VV = np.meshgrid(u_s, v_s)
    xE = (R_E/1000.0)*np.cos(UU)*np.sin(VV)
    yE = (R_E/1000.0)*np.sin(UU)*np.sin(VV)
    zE = (R_E/1000.0)*np.cos(VV)
    fig3d.add_trace(go.Surface(x=xE, y=yE, z=zE, opacity=0.35, showscale=False, name="Earth"))
    # trace 1: Sunlit line
    fig3d.add_trace(go.Scatter3d(x=x_sun, y=y_sun, z=z_sun, mode="lines",
                                 line=dict(color="gold", width=4), name="Sunlit"))
    # trace 2: Eclipse line
    fig3d.add_trace(go.Scatter3d(x=x_ecl, y=y_ecl, z=z_ecl, mode="lines",
                                 line=dict(color="gray", width=4), name="Eclipse"))
    # trace 3: Sat marker (animated)
    fig3d.add_trace(go.Scatter3d(x=[x_km[0]], y=[y_km[0]], z=[z_km[0]],
                                 mode="markers", marker=dict(size=6, color="red"),
                                 name="Sat"))

    # Frames update only trace 3 so Earth/lines persist
    frames3d = []
    step = 4
    for k in range(0, len(x_km), step):
        frames3d.append(go.Frame(
            name=str(k),
            data=[go.Scatter3d(x=[x_km[k]], y=[y_km[k]], z=[z_km[k]],
                               mode="markers", marker=dict(size=6, color="red"))],
            traces=[3]
        ))
    fig3d.frames = frames3d
    fig3d.update_layout(scene=dict(aspectmode="data"),
                        height=560, showlegend=True,
                        margin=dict(l=0, r=0, t=0, b=0),
                        uirevision="keep-earth")
    if show_play:
        fig3d.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05, x=0.06,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=int(40/anim_speed), redraw=True),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                )]
            )]
        )
    st.plotly_chart(fig3d, use_container_width=True)

    # Ground track figure with synchronized marker
    fig_gt = go.Figure()
    fig_gt.add_trace(go.Scattergeo(lon=lon_deg, lat=lat_deg, mode="lines",
                                   line=dict(color="royalblue", width=2), name="Path"))  # trace 0
    fig_gt.add_trace(go.Scattergeo(lon=[lon_deg[0]], lat=[lat_deg[0]], mode="markers",
                                   marker=dict(size=6, color="red"), name="Sat"))        # trace 1
    frames_gt = []
    for k in range(0, len(lon_deg), step):
        frames_gt.append(go.Frame(
            name=str(k),
            data=[go.Scattergeo(lon=[lon_deg[k]], lat=[lat_deg[k]], mode="markers",
                                marker=dict(size=6, color="red"))],
            traces=[1]
        ))
    fig_gt.frames = frames_gt
    fig_gt.update_layout(geo=dict(projection_type="natural earth", showland=True),
                         height=360, margin=dict(t=10), uirevision="keep-gt")
    if show_play:
        fig_gt.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.02, x=0.02,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=int(40/anim_speed), redraw=True),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                )]
            )]
        )
    st.plotly_chart(fig_gt, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Orbital period (min)", f"{sim.T_orbit/60.0:.2f}")
    c2.metric("Eclipse fraction", f"{sim.eclipse_fraction():.3f}")
    c3.metric("Earth view factor", f"{sim.VF:.3f}")

# =========================
# TAB 2: POWER
# =========================
with tab_power:
    st.subheader("Power — instantaneous, average, and SoC")
    t, u, x_km, y_km, z_km = sim.orbit_eci(N=720)
    P_inst, cos_inc, ecl = sim.instantaneous_power(attitude, t, x_km, y_km, z_km,
                                                   sim.A_panel, sim.eta)
    P_inst = P_inst * cal_factor
    P_avg = float(P_inst.mean())
    daily_gen_Wh = P_avg * 86400.0 / 3600.0

    cons_W = st.slider("Average consumption (W)", 0.1, 20.0, 3.0, 0.1)
    daily_cons_Wh = cons_W * 24.0

    batt_Wh = st.slider("Battery capacity (Wh)", 5.0, 500.0, 30.0, 1.0)
    start_soc = st.slider("Start SoC (%)", 0, 100, 50)

    net_daily = daily_gen_Wh - daily_cons_Wh
    days = np.arange(1, mission_days+1)
    soc_list, soc_wh = [], start_soc/100.0 * batt_Wh
    for _ in days:
        soc_wh = min(max(soc_wh + net_daily, 0.0), batt_Wh)
        soc_list.append(100.0 * soc_wh / batt_Wh)
    df_soc = pd.DataFrame({"Day": days, "SoC (%)": soc_list})

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg generation (W)", f"{P_avg:.3f}")
    c2.metric("Daily generation (Wh)", f"{daily_gen_Wh:.1f}")
    c3.metric("Daily consumption (Wh)", f"{daily_cons_Wh:.1f}")

    figP = px.line(x=(t/t[-1])*360.0, y=P_inst,
                   labels={"x":"Orbit phase (deg)", "y":"Power (W)"},
                   title=f"Instantaneous Power — {attitude} (cal factor {cal_factor:.2f})")
    st.plotly_chart(figP, use_container_width=True)
    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True,
                            title="Battery SoC over mission"),
                    use_container_width=True)
    if df_soc["SoC (%)"].iloc[-1] < 20.0:
        st.warning("Projected final SoC < 20% — consider more panel area/efficiency, better pointing, or lower load.")
    else:
        st.success("Battery SoC projection looks acceptable.")

# =========================
# TAB 3: THERMAL
# =========================
with tab_thermal:
    st.subheader("Radiative Thermal Equilibrium (averaged)")
    A_abs = st.number_input("Absorbing face area (m²)", 0.001, 2.0, sim.A_panel, 0.001)
    A_rad = st.number_input("Radiating area (m²)", 0.01, 6.0, 6.0*sim.A_panel, 0.01)
    T_c, Qs, Qa, Qir, Qabs = sim.thermal_equilibrium(A_abs=A_abs, A_rad=A_rad)
    st.metric("Equilibrium temperature (°C)", f"{T_c:.2f}")
    dfQ = pd.DataFrame([{"Solar_avg_W": Qs, "Albedo_W": Qa, "Earth_IR_W": Qir, "Total_abs_W": Qabs}])
    st.dataframe(dfQ, use_container_width=True)
    st.plotly_chart(px.bar(dfQ.melt(var_name="Component", value_name="W"),
                           x="Component", y="W", title="Absorbed power components"),
                    use_container_width=True)

# =========================
# TAB 4: DRAG
# =========================
with tab_drag:
    st.subheader("Altitude Decay from Drag (simple model)")
    A_drag = st.number_input("Reference drag area (m²)", 0.001, 2.0, sim.A_panel, 0.001)
    alt_series = sim.drag_decay_days(mission_days, A_drag=A_drag)
    df_alt = pd.DataFrame({"Day": np.arange(1, mission_days+1), "Altitude (km)": alt_series})
    st.plotly_chart(px.line(df_alt, x="Day", y="Altitude (km)", markers=True,
                            title="Altitude decay over mission"),
                    use_container_width=True)

# =========================
# TAB 5: SAVE/LOAD & EXPORT (Pro-only)
# =========================
with tab_io:
    st.subheader("Save / Load Missions & Export Data")
    if st.session_state.plan != "Pro":
        st.info("🔒 This feature is available on the **Pro plan ($9/mo)**.")
        st.write("- Save mission parameters to JSON")
        st.write("- Load mission JSON")
        st.write("- Export Orbit CSV and Power CSV")
        st.write("")
        st.write("Upgrade in the sidebar to unlock.")
        st.stop()

    # ---- Save parameters to JSON (download) ----
    mission_params = {
        "altitude_km": altitude_km, "incl_deg": incl_deg,
        "mass_kg": mass_kg, "Cd": Cd,
        "panel_area_m2": panel_area, "panel_eff": panel_eff,
        "absorptivity": absorp, "emissivity": emiss,
        "attitude": attitude, "auto_cal": auto_cal
    }
    json_bytes = json.dumps(mission_params, indent=2).encode("utf-8")
    st.download_button("Download Mission JSON", data=json_bytes,
                       file_name="mission.json", mime="application/json")

    # ---- Load parameters from JSON (upload) ----
    up = st.file_uploader("Load Mission JSON", type=["json"])
    if up is not None:
        try:
            data = json.load(up)
            st.write("Loaded mission:", data)
            st.info("Apply these values by re-setting the sidebar controls.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    # ---- Export orbit & power CSVs ----
    t, u, x_km, y_km, z_km = sim.orbit_eci(N=720)
    lon_deg, lat_deg = sim.ground_track_from_eci(t, x_km, y_km, z_km)
    P_inst, cos_inc, ecl = sim.instantaneous_power(attitude, t, x_km, y_km, z_km, sim.A_panel, sim.eta)
    P_inst = P_inst * cal_factor

    df_orbit = pd.DataFrame({
        "t_sec": t,
        "x_eci_km": x_km, "y_eci_km": y_km, "z_eci_km": z_km,
        "lon_deg": lon_deg, "lat_deg": lat_deg,
        "eclipsed": ecl.astype(int)
    })
    df_power = pd.DataFrame({
        "t_sec": t, "power_W": P_inst, "cos_incidence": cos_inc, "eclipsed": ecl.astype(int)
    })

    buf_orbit = io.StringIO(); df_orbit.to_csv(buf_orbit, index=False)
    buf_power = io.StringIO(); df_power.to_csv(buf_power, index=False)
    st.download_button("Export Orbit CSV", buf_orbit.getvalue(), file_name="orbit.csv", mime="text/csv")
    st.download_button("Export Power CSV", buf_power.getvalue(), file_name="power.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Phase-1: public simulator with GeneSat-1 preset & optional power calibration; aligned ground track via synchronized ECI→ECEF; "
    "attitude-dependent power; Stefan–Boltzmann thermal; simple drag decay; and **Pro-gated Save/Load/Export**. "
    "Wire Stripe & Supabase in Phase-2."
)
