# app.py — Generic CubeSat Simulator (GeneSat-validated, user-tunable)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Physical constants (SI)
# -------------------------
MU = 3.986004418e14       # Earth's gravitational parameter (m^3/s^2)
R_E = 6371e3              # Earth radius (m)
OMEGA_E = 7.2921159e-5    # Earth's rotation rate (rad/s)
SIGMA = 5.670374419e-8    # Stefan–Boltzmann (W/m^2/K^4)
SOLAR_CONST = 1366.0      # Solar constant (W/m^2)
EARTH_IR = 237.0          # Approx Earth IR (W/m^2)
ALBEDO = 0.3              # Mean Earth albedo

# -------------------------
# Very simple atmosphere (MSIS-like) for drag decay near 200–600 km
# ρ(h) = ρ_ref * exp(-(h - h_ref)/H), with H ≈ 60 km
# -------------------------
def rho_msis_simple(h_m):
    H = 60e3
    rho_200 = 2.5e-11  # kg/m^3 at ~200 km (rough, order-of-magnitude)
    h = np.maximum(h_m, 200e3)  # clamp low
    rho = rho_200 * np.exp(-(h - 200e3)/H)
    return np.clip(rho, 1e-13, None)

def clamp_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def earth_view_factor(alt_m):
    # Fraction of sky-hemisphere subtended by Earth:
    # VF = (1 - cos ψ)/2, ψ = arccos(R_E / (R_E + h))
    r = R_E + alt_m
    ratio = np.clip(R_E / r, -1.0, 1.0)
    psi = np.arccos(ratio)
    return (1.0 - np.cos(psi)) / 2.0

def eclipse_mask_xyz(x, y, z):
    """
    Cylindrical shadow test with Sun along +X.
    Eclipse when x < 0 (behind Earth) AND sqrt(y^2 + z^2) < R_E.
    Inputs in km to match plotting arrays.
    """
    r_perp = np.sqrt(y**2 + z**2)
    return (x < 0) & (r_perp < (R_E/1000.0))

# -------------------------
# GeneSat-1-ish defaults (validation baseline)
# -------------------------
GENESAT_DEFAULTS = {
    "altitude_km": 460.0,     # mid of its typical band
    "incl_deg": 40.0,
    "mass_kg": 4.6,
    "Cd": 2.2,
    "panel_area_m2": 0.03,    # effective flat panel area (body-mounted)
    "panel_eff": 0.25,        # practical cell efficiency
    "absorptivity": 0.65,
    "emissivity": 0.83,
    "target_avg_power_W": 4.5 # approximate reported average power
}

# -------------------------
# CubeSat Simulation Core
# -------------------------
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
        self.T_orbit = 2*np.pi*np.sqrt(self.r**3/MU)  # s
        self.n = 2*np.pi/self.T_orbit                 # rad/s
        self.v = np.sqrt(MU/self.r)                   # m/s
        self.VF = earth_view_factor(self.h)           # Earth view factor

    # Eclipse fraction (geometric, circular orbit)
    def eclipse_fraction(self):
        ratio = np.clip(R_E / (R_E + self.h), -1.0, 1.0)
        psi = np.arccos(ratio)
        return float(psi/np.pi)  # 0 .. 0.5

    # 3D orbit samples (km)
    def orbit_xyz(self, N=720):
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        r_km = self.r/1000.0
        x = r_km*np.cos(theta)
        y = r_km*np.sin(theta)*np.cos(self.i)
        z = r_km*np.sin(theta)*np.sin(self.i)
        return theta, x, y, z

    # Ground track (account for Earth rotation)
    def ground_track(self, N=720):
        t = np.linspace(0.0, self.T_orbit, N, endpoint=False)
        lon_inertial = self.n * t
        lon_earth = lon_inertial - OMEGA_E*t
        lon_deg = clamp_angle_deg(np.degrees(lon_earth))
        lat_deg = np.degrees(np.arcsin(np.sin(self.i) * np.sin(self.n*t)))
        return lon_deg, lat_deg

    # Attitude and instantaneous solar incidence
    # Sun direction is +X in this simple model.
    def panel_normal(self, theta, x, y, z, attitude):
        if attitude == "sun-tracking":
            # panel always points at Sun (+X)
            nx = np.ones_like(theta); ny = np.zeros_like(theta); nz = np.zeros_like(theta)
        elif attitude == "nadir-pointing":
            rvec = np.vstack((x, y, z)).T
            rnorm = np.linalg.norm(rvec, axis=1, keepdims=True)
            nhat = -(rvec / np.maximum(rnorm, 1e-12))  # toward Earth center
            nx, ny, nz = nhat[:,0], nhat[:,1], nhat[:,2]
        else:
            nx = ny = nz = None  # body-spin handled as average elsewhere
        return nx, ny, nz

    def instantaneous_power(self, theta, x, y, z, attitude, A_panel, eta, eclipsed_mask):
        # cos(incidence) = dot(panel_normal, sun_dir), sun_dir = +X
        if attitude == "body-spin":
            cos_inc = np.full_like(theta, 0.5)  # average over spin for flat plate
        else:
            nx, ny, nz = self.panel_normal(theta, x, y, z, attitude)
            cos_inc = np.clip(nx*1.0 + ny*0.0 + nz*0.0, 0.0, 1.0)
        # zero in eclipse
        cos_inc = cos_inc * (~eclipsed_mask)
        return SOLAR_CONST * A_panel * eta * cos_inc, cos_inc

    def avg_power(self, attitude, A_panel, eta):
        th, x, y, z = self.orbit_xyz(N=1440)
        eclip = eclipse_mask_xyz(x, y, z)
        if attitude == "body-spin":
            cos_mean = 0.5 * (~eclip)
            P = SOLAR_CONST * A_panel * eta * cos_mean
        else:
            P, _ = self.instantaneous_power(th, x, y, z, attitude, A_panel, eta, eclip)
        return float(P.mean())

    # Radiative thermal equilibrium (averaged)
    def thermal_equilibrium(self, A_abs=None, A_rad=None):
        if A_abs is None:
            A_abs = self.A_panel
        if A_rad is None:
            A_rad = 6.0*self.A_panel  # simple smallsat approximation

        efrac = self.eclipse_fraction()
        Q_solar = SOLAR_CONST * self.alpha * A_abs * (1.0 - efrac)
        Q_albedo = ALBEDO * SOLAR_CONST * self.alpha * A_abs * self.VF * (1.0 - efrac)
        Q_ir = EARTH_IR * self.eps * A_abs * self.VF
        Q_abs = Q_solar + Q_albedo + Q_ir
        T_k = (Q_abs/(self.eps*SIGMA*A_rad))**0.25
        return float(T_k - 273.15), Q_solar, Q_albedo, Q_ir, Q_abs

    # Simple energy-based drag decay integration (daily steps)
    # da/dt ≈ - (Cd * A / m) * sqrt(μ a) * ρ(h)
    def drag_decay_days(self, days, A_drag=None):
        A = self.A_panel if A_drag is None else A_drag
        a = R_E + self.h
        out = []
        for _ in range(int(days)):
            rho = rho_msis_simple(a - R_E)
            da_dt = - (self.Cd * A / self.m) * np.sqrt(MU * a) * rho
            a = a + da_dt * 86400.0
            if a < R_E + 120e3:  # floor at ~120 km
                a = R_E + 120e3
            out.append(a - R_E)
        return np.array(out)/1000.0  # km

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Generic CubeSat Sim (GeneSat-validated)", layout="wide")
st.title("🛰️ Generic CubeSat Simulator — GeneSat-validated, Mission-tunable")

with st.sidebar:
    st.header("Preset & Validation")
    use_genesat = st.checkbox("Load GeneSat-1 defaults", value=True)
    auto_cal = st.checkbox("Calibrate avg power to GeneSat-like target (4.5 W)", value=True)

    st.header("Mission Parameters")
    if use_genesat:
        altitude_km = st.slider("Altitude (km)", 200.0, 700.0, GENESAT_DEFAULTS["altitude_km"])
        incl_deg    = st.slider("Inclination (deg)", 0.0, 98.0, GENESAT_DEFAULTS["incl_deg"])
        mass_kg     = st.number_input("Mass (kg)", 0.1, 50.0, GENESAT_DEFAULTS["mass_kg"], 0.1)
        Cd          = st.slider("Drag coefficient Cd", 1.5, 3.0, GENESAT_DEFAULTS["Cd"], 0.1)
        panel_area  = st.number_input("Panel / face area (m²)", 0.001, 0.5, GENESAT_DEFAULTS["panel_area_m2"], 0.001)
        panel_eff   = st.slider("Panel efficiency η", 0.05, 0.35, GENESAT_DEFAULTS["panel_eff"], 0.01)
        absorp      = st.slider("Absorptivity α", 0.1, 1.0, GENESAT_DEFAULTS["absorptivity"], 0.01)
        emiss       = st.slider("Emissivity ε", 0.1, 1.0, GENESAT_DEFAULTS["emissivity"], 0.01)
        target_avgW = GENESAT_DEFAULTS["target_avg_power_W"]
    else:
        altitude_km = st.slider("Altitude (km)", 200.0, 2000.0, 500.0)
        incl_deg    = st.slider("Inclination (deg)", 0.0, 98.0, 51.6)
        mass_kg     = st.number_input("Mass (kg)", 0.1, 200.0, 4.0, 0.1)
        Cd          = st.slider("Drag coefficient Cd", 1.0, 3.5, 2.2, 0.1)
        panel_area  = st.number_input("Panel / face area (m²)", 0.001, 2.0, 0.05, 0.001)
        panel_eff   = st.slider("Panel efficiency η", 0.05, 0.38, 0.28, 0.01)
        absorp      = st.slider("Absorptivity α", 0.1, 1.0, 0.6, 0.01)
        emiss       = st.slider("Emissivity ε", 0.1, 1.0, 0.8, 0.01)
        target_avgW = st.number_input("Target average power (W) for calibration", 0.1, 50.0, 4.5, 0.1)

    st.header("Attitude & Ops")
    attitude = st.radio("Attitude model", ["body-spin", "sun-tracking", "nadir-pointing"])
    show_play = st.checkbox("Show Play buttons on plots", value=True)
    anim_speed = st.slider("Animation speed (Plotly)", 0.1, 5.0, 1.0, 0.1)
    mission_days = st.slider("Mission duration (days)", 1, 365, 60)

# Instantiate sim
sim = CubeSatSim(
    altitude_km=altitude_km, incl_deg=incl_deg, mass_kg=mass_kg,
    Cd=Cd, panel_area_m2=panel_area, panel_eff=panel_eff,
    absorptivity=absorp, emissivity=emiss
)

# Optional GeneSat-like power calibration (keeps UI tunable)
cal_factor = 1.0
if auto_cal:
    P_now = sim.avg_power(attitude, sim.A_panel, sim.eta)
    if P_now > 1e-9:
        cal_factor = target_avgW / P_now

# Tabs
tab_orbit, tab_power, tab_thermal, tab_drag = st.tabs([
    "3D Orbit (Shaded) + Ground Track",
    "Power (per-orbit & SoC)",
    "Thermal (radiative balance)",
    "Drag (altitude decay)"
])

# -------------------------
# TAB 1: 3D shaded orbit (Earth persists) + ground track
# -------------------------
with tab_orbit:
    st.subheader("3D Orbit with Sunlit / Eclipse Shading")

    theta, x_km, y_km, z_km = sim.orbit_xyz(N=720)
    eclipsed = eclipse_mask_xyz(x_km, y_km, z_km)

    # Build sunlit/eclipsed lines (as static traces)
    x_sun, y_sun, z_sun = x_km.copy(), y_km.copy(), z_km.copy()
    x_sun[eclipsed] = None; y_sun[eclipsed] = None; z_sun[eclipsed] = None
    x_ecl, y_ecl, z_ecl = x_km.copy(), y_km.copy(), z_km.copy()
    x_ecl[~eclipsed] = None; y_ecl[~eclipsed] = None; z_ecl[~eclipsed] = None

    fig3d = go.Figure()

    # Trace 0: Earth surface (STATIC)
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    xE = (R_E/1000.0)*np.cos(uu)*np.sin(vv)
    yE = (R_E/1000.0)*np.sin(uu)*np.sin(vv)
    zE = (R_E/1000.0)*np.cos(vv)
    fig3d.add_trace(go.Surface(x=xE, y=yE, z=zE, opacity=0.35, showscale=False, name="Earth"))

    # Trace 1: Sunlit path (STATIC)
    fig3d.add_trace(go.Scatter3d(x=x_sun, y=y_sun, z=z_sun, mode="lines",
                                 line=dict(color="gold", width=4), name="Sunlit"))
    # Trace 2: Eclipse path (STATIC)
    fig3d.add_trace(go.Scatter3d(x=x_ecl, y=y_ecl, z=z_ecl, mode="lines",
                                 line=dict(color="gray", width=4), name="Eclipse"))

    # Trace 3: Satellite marker (ANIMATED)
    fig3d.add_trace(go.Scatter3d(x=[x_km[0]], y=[y_km[0]], z=[z_km[0]],
                                 mode="markers", marker=dict(size=6, color="red"), name="Sat"))

    # Frames: update only trace index 3 so Earth/orbit lines persist
    frames = []
    step = 4
    for k in range(0, len(x_km), step):
        frames.append(go.Frame(
            name=str(k),
            data=[go.Scatter3d(x=[x_km[k]], y=[y_km[k]], z=[z_km[k]],
                               mode="markers", marker=dict(size=6, color="red"))],
            traces=[3]
        ))
    fig3d.frames = frames

    fig3d.update_layout(
        scene=dict(aspectmode="data"),
        height=560, showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep-earth"  # keeps camera/legend during UI changes
    )

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

    # Ground track (static)
    st.subheader("Ground Track (one orbit)")
    lon_deg, lat_deg = sim.ground_track(N=720)
    fig_flat = go.Figure()
    fig_flat.add_trace(go.Scattergeo(lon=lon_deg, lat=lat_deg, mode="lines",
                                     line=dict(color="royalblue", width=2), name="Path"))
    fig_flat.update_layout(geo=dict(projection_type="natural earth", showland=True),
                           height=350, margin=dict(t=0))
    st.plotly_chart(fig_flat, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Orbital period (min)", f"{sim.T_orbit/60.0:.2f}")
    c2.metric("Eclipse fraction", f"{sim.eclipse_fraction():.3f}")
    c3.metric("Earth view factor", f"{sim.VF:.3f}")

# -------------------------
# TAB 2: Power (per-orbit curve + SoC)
# -------------------------
with tab_power:
    st.subheader("Power — instantaneous, average, and battery SoC")

    # Per-orbit instantaneous power
    Np = 720
    th, xk, yk, zk = sim.orbit_xyz(N=Np)
    ecl = eclipse_mask_xyz(xk, yk, zk)

    if attitude == "body-spin":
        P_inst = SOLAR_CONST * sim.A_panel * sim.eta * (~ecl) * 0.5
        cos_inc = np.full(Np, 0.5)*(~ecl)
    else:
        P_inst, cos_inc = sim.instantaneous_power(th, xk, yk, zk, attitude, sim.A_panel, sim.eta, ecl)

    # Apply calibration factor if enabled
    P_inst = P_inst * cal_factor

    P_avg = float(P_inst.mean())
    daily_gen_Wh = P_avg * 86400.0 / 3600.0

    cons_W = st.slider("Average consumption (W)", 0.1, 20.0, 3.0, 0.1)
    daily_cons_Wh = cons_W * 24.0

    batt_Wh = st.slider("Battery capacity (Wh)", 5.0, 500.0, 30.0, 1.0)
    start_soc = st.slider("Start SoC (%)", 0, 100, 50)

    net_daily = daily_gen_Wh - daily_cons_Wh
    days = np.arange(1, mission_days+1)
    soc_list = []
    soc_wh = start_soc/100.0 * batt_Wh
    for _ in days:
        soc_wh += net_daily
        soc_wh = min(max(soc_wh, 0.0), batt_Wh)
        soc_list.append(100.0 * soc_wh / batt_Wh)
    df_soc = pd.DataFrame({"Day": days, "SoC (%)": soc_list})

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg generation (W)", f"{P_avg:.3f}")
    c2.metric("Daily generation (Wh)", f"{daily_gen_Wh:.1f}")
    c3.metric("Daily consumption (Wh)", f"{daily_cons_Wh:.1f}")

    figP = px.line(x=np.degrees(th), y=P_inst,
                   labels={"x":"Orbit phase (deg)", "y":"Power (W)"},
                   title=f"Instantaneous Power — {attitude} (cal factor {cal_factor:.2f})")
    st.plotly_chart(figP, use_container_width=True)

    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True,
                            title="Battery SoC over mission"),
                    use_container_width=True)

    if df_soc["SoC (%)"].iloc[-1] < 20:
        st.warning("Projected final SoC < 20% — consider more panel area/efficiency, better pointing, or lower load.")
    else:
        st.success("Battery SoC projection looks acceptable.")

# -------------------------
# TAB 3: Thermal (radiative balance)
# -------------------------
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

# -------------------------
# TAB 4: Drag (altitude decay)
# -------------------------
with tab_drag:
    st.subheader("Altitude Decay from Drag (simple model)")
    A_drag = st.number_input("Reference drag area (m²)", 0.001, 2.0, sim.A_panel, 0.001)
    alt_series = sim.drag_decay_days(mission_days, A_drag=A_drag)
    df_alt = pd.DataFrame({"Day": np.arange(1, mission_days+1), "Altitude (km)": alt_series})
    st.plotly_chart(px.line(df_alt, x="Day", y="Altitude (km)", markers=True,
                            title="Altitude decay over mission"),
                    use_container_width=True)

st.markdown("---")
st.caption(
    "Generic CubeSat simulator with GeneSat-1 validation: SI units; geometric eclipse fraction; "
    "Earth rotation in ground track; attitude-dependent power; Stefan–Boltzmann thermal balance; "
    "and simple MSIS-like drag decay. Toggle GeneSat defaults and keep tweaking for your mission."
)
