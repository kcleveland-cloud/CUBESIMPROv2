# app.py — CubeSat Simulator (β-angle sweep, GeneSat-validated, mission-tunable, Pro-gated I/O)
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
ALBEDO = 0.3              # dimensionless

# =========================
# Helpers
# =========================
def clamp_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def earth_view_factor(alt_m):
    """
    Fraction of the sky hemisphere subtended by Earth, for altitude h.
    Angular radius of Earth's disk: theta = arcsin(Re / r).
    VF = (1 - cos(theta)) / 2
    """
    r = R_E + float(alt_m)
    theta = np.arcsin(np.clip(R_E / r, -1.0, 1.0))
    return (1.0 - np.cos(theta)) / 2.0

def rho_msis_simple(h_m):
    # Very simple exponential atmosphere ~200–600 km band
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

# ---- Sun vector and eclipse with β-angle ----
def sun_vector_eci(incl_rad, beta_deg):
    """
    Build a fixed Sun unit vector S in ECI given orbit inclination and β.
    - ĥ is orbit-normal in ECI for our parameterization.
    - β is the angle between Sun vector and orbital plane (positive toward +ĥ).
    We align Sun's in-plane projection with +X ECI (line of nodes) for simplicity.
    """
    hhat = np.array([0.0, -np.sin(incl_rad), np.cos(incl_rad)])   # orbit normal (unit)
    cb = np.cos(np.radians(beta_deg)); sb = np.sin(np.radians(beta_deg))
    S = cb * np.array([1.0, 0.0, 0.0]) + sb * hhat
    S /= np.linalg.norm(S)
    return S

def eclipse_mask_from_vec(x_km, y_km, z_km, S_vec):
    """
    Cylindrical shadow test with arbitrary Sun axis S_vec (unit).
    ECI position r (km) is in umbra if:
        r·S < 0   and   ||r - (r·S)S|| < R_E (in km)
    Vectorized for arrays.
    """
    r = np.vstack((x_km, y_km, z_km)).T          # [N,3] km
    S = np.asarray(S_vec, dtype=float)           # [3]
    r_dot_S = r @ S                               # [N]
    r_sq = np.sum(r*r, axis=1)                    # [N]
    dist_ax_sq = r_sq - r_dot_S**2                # [N]
    return (r_dot_S < 0.0) & (dist_ax_sq < (R_E/1000.0)**2)

def eclipse_fraction_beta(alt_m, beta_deg):
    """
    Orbit-averaged eclipse fraction for circular orbit, cylindrical shadow with β.
    f = (1/π) * arccos( sqrt(1 - (Re/r)^2) / cos β ),    if cosβ>0 and value<1
      = 0,                                               otherwise
    Reduces to f = (1/π) * asin(Re/r) at β = 0.
    """
    r = R_E + float(alt_m)
    Re_r = np.clip(R_E / r, 0.0, 1.0)
    root = np.sqrt(1.0 - Re_r**2)               # = cos(asin(Re/r))
    cb = np.cos(np.radians(beta_deg))
    if cb <= 0.0:
        return 0.0
    val = root / cb
    if val >= 1.0:
        return 0.0
    return float(np.arccos(val) / np.pi)

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
    def __init__(self, altitude_km, incl_deg, mass_kg, Cd, panel_area_m2, panel_eff, absorptivity, emissivity, beta_deg=0.0):
        self.alt_km = float(altitude_km)
        self.h = self.alt_km * 1000.0
        self.i = np.radians(float(incl_deg))
        self.beta_deg = float(beta_deg)

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
        self.VF = earth_view_factor(self.h)            # Earth view factor

        # Sun direction for this orbit (unit, ECI)
        self.S = sun_vector_eci(self.i, self.beta_deg)

    def set_beta(self, beta_deg):
        self.beta_deg = float(beta_deg)
        self.S = sun_vector_eci(self.i, self.beta_deg)

    def eclipse_fraction(self):
        """β-aware cylindrical eclipse fraction."""
        return eclipse_fraction_beta(self.h, self.beta_deg)

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

    # ---------- Attitude & power (uses Sun vector S) ----------
    def instantaneous_power(self, attitude, t, x_km, y_km, z_km, A_panel, eta):
        S = self.S
        r = np.vstack((x_km, y_km, z_km)).T
        rnorm = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / np.maximum(rnorm, 1e-12)

        if attitude == "body-spin":
            cos_inc = np.full_like(t, 0.5)
        elif attitude == "sun-tracking":
            cos_inc = np.ones_like(t)  # an actively sun-pointing panel
        elif attitude == "nadir-pointing":
            nhat = -rhat
            cos_inc = nhat @ S  # dot with Sun vector
        else:
            cos_inc = np.full_like(t, 0.5)

        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        ecl = eclipse_mask_from_vec(x_km, y_km, z_km, S)
        cos_inc = cos_inc * (~ecl)
        P = SOLAR_CONST * A_panel * eta * cos_inc
        return P, cos_inc, ecl

    def avg_power(self, attitude, A_panel, eta):
        t, u, x, y, z = self.orbit_eci(N=1440)
        P, _, _ = self.instantaneous_power(attitude, t, x, y, z, A_panel, eta)
        return float(P.mean())

    def avg_power_at_beta(self, attitude, beta_deg, A_panel, eta, N=720):
        """Orbit-average power for a given β without mutating state outside."""
        S = sun_vector_eci(self.i, beta_deg)
        t, u, x, y, z = self.orbit_eci(N=N)
        r = np.vstack((x, y, z)).T
        rnorm = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = r / np.maximum(rnorm, 1e-12)

        if attitude == "body-spin":
            cos_inc = np.full_like(t, 0.5)
        elif attitude == "sun-tracking":
            cos_inc = np.ones_like(t)
        elif attitude == "nadir-pointing":
            nhat = -rhat
            cos_inc = nhat @ S
        else:
            cos_inc = np.full_like(t, 0.5)

        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        ecl = eclipse_mask_from_vec(x, y, z, S)
        cos_inc = cos_inc * (~ecl)
        P = SOLAR_CONST * A_panel * eta * cos_inc
        return float(P.mean())

    # ---------- Thermal (corrected VF + internal dissipation, β-aware eclipse) ----------
    def thermal_equilibrium(self, A_abs=None, A_rad=None, Q_internal_W=0.0):
        """
        Radiative equilibrium (orbit-averaged):
            Q_solar_avg + Q_albedo_avg + Q_IR + Q_internal = ε σ A_rad T^4
        A_abs : absorbing area for solar/albedo/IR (m^2)
        A_rad : effective radiating area (m^2)
        Q_internal_W : electronics heat (W), e.g., average bus dissipation
        """
        if A_abs is None:
            A_abs = self.A_panel
        if A_rad is None:
            A_rad = 6.0 * self.A_panel  # default; user should tune

        efrac = self.eclipse_fraction()
        VF = self.VF  # Earth view factor independent of β in this simple model

        Q_solar = SOLAR_CONST * self.alpha * A_abs * (1.0 - efrac)
        Q_albedo = ALBEDO * SOLAR_CONST * self.alpha * A_abs * VF * (1.0 - efrac)
        # For IR, use emissivity as absorptivity (Kirchhoff)
        Q_ir = EARTH_IR * self.eps * A_abs * VF
        Q_total = Q_solar + Q_albedo + Q_ir + float(Q_internal_W)

        T_K = (Q_total / (self.eps * SIGMA * A_rad)) ** 0.25
        return float(T_K - 273.15), Q_solar, Q_albedo, Q_ir, float(Q_internal_W), float(Q_total)

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
st.title("🛰️ CubeSat Simulator — Phase 1 (β-angle, GeneSat-validated, mission-tunable)")

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
            st.session_state.plan = "Free"   # <-- fixed (removed stray ')')

    st.header("Plan")
    st.markdown(f"**Current plan:** {st.session_state.plan}")
    if st.session_state.plan == "Free":
        if st.button("Upgrade to Pro ($9/mo)"):
            st.session_state.plan = "Pro"  # Phase-1 stub; wire to Stripe later
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
    elec_derate = st.slider("Electrical derate (BOL→EOL, MPPT, wiring)", 0.40, 1.00, 0.70, 0.01)
    beta_deg = st.slider("β-angle (deg) — Sun vs. orbital plane", -80.0, 80.0, 0.0, 0.5)
    show_play = st.checkbox("Show Play buttons on plots", True)
    anim_speed = st.slider("Animation speed (Plotly)", 0.1, 5.0, 1.0, 0.1)
    mission_days = st.slider("Mission duration (days)", 1, 365, 60)

# Build simulator & optional power calibration
sim = CubeSatSim(
    altitude_km=altitude_km, incl_deg=incl_deg, mass_kg=mass_kg,
    Cd=Cd, panel_area_m2=panel_area, panel_eff=panel_eff,
    absorptivity=absorp, emissivity=emiss, beta_deg=beta_deg
)
sim.set_beta(beta_deg)

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
    eclipsed = eclipse_mask_from_vec(x_km, y_km, z_km, sim.S)

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
                direction="left",
                showactive=False,
                x=0.05, xanchor="left",
                y=0.05, yanchor="bottom",   # inside bottom-left
                pad={"r": 0, "t": 0},
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
                direction="left",
                showactive=False,
                x=0.05, xanchor="left",
                y=0.05, yanchor="bottom",   # inside bottom-left
                pad={"r": 0, "t": 0},
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
    st.caption("Tip: increasing |β| shortens/eradicates eclipse; beyond a critical β there's no eclipse.")

# =========================
# TAB 2: POWER
# =========================
with tab_power:
    st.subheader("Power — instantaneous, average, β-sweep, and SoC")
    st.caption("**Power Reality Check:** Orbit-average power (OAP) is often 30–60% of panel peak. "
               "Benchmarks: GeneSat-1 ≈ 4–5 W OAP (3U body-mounted), typical 1U body-mounted ≈ 1–2 W OAP. "
               "Vendor specs are peak, not orbit-average.")

    t, u, x_km, y_km, z_km = sim.orbit_eci(N=720)
    P_inst, cos_inc, ecl = sim.instantaneous_power(attitude, t, x_km, y_km, z_km,
                                                   sim.A_panel, sim.eta)
    # Apply calibration toward target (for presets) + electrical derate to reconcile peak → OAP
    P_inst = P_inst * cal_factor * elec_derate

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

    # Instantaneous power plot
    figP = px.line(x=(t/t[-1])*360.0, y=P_inst,
                   labels={"x":"Orbit phase (deg)", "y":"Power (W)"},
                   title=f"Instantaneous Power — {attitude} (cal {cal_factor:.2f} × derate {elec_derate:.2f} @ β={beta_deg:.1f}°)")
    st.plotly_chart(figP, use_container_width=True)

    # -------- β-angle sweep (OAP vs β) --------
    st.markdown("### β-angle sweep: Orbit-average power vs β")
    betas = np.linspace(-80, 80, 161)
    oap = np.array([sim.avg_power_at_beta(attitude, b, sim.A_panel, sim.eta) for b in betas])
    # Apply same scaling as instantaneous stream
    oap_scaled = oap * cal_factor * elec_derate
    df_beta = pd.DataFrame({"beta_deg": betas, "OAP_W": oap_scaled})
    fig_beta = px.line(df_beta, x="beta_deg", y="OAP_W",
                       title="Orbit-average Power vs β-angle",
                       labels={"beta_deg":"β (deg)", "OAP_W":"Orbit-average Power (W)"})
    st.plotly_chart(fig_beta, use_container_width=True)

    # Battery SoC projection
    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True,
                            title="Battery SoC over mission"),
                    use_container_width=True)
    if df_soc["SoC (%)"].iloc[-1] < 20.0:
        st.warning("Projected final SoC < 20% — consider more panel area/efficiency, better pointing, or lower load.")
    else:
        st.success("Battery SoC projection looks acceptable.")

# =========================
# TAB 3: THERMAL (corrected)
# =========================
with tab_thermal:
    st.subheader("Radiative Thermal Equilibrium (averaged) — corrected VF & internal heat")
    A_abs = st.number_input("Absorbing area A_abs (m²)", 0.001, 2.0, sim.A_panel, 0.001)
    A_rad = st.number_input("Radiating area A_rad (m²)", 0.005, 2.0, 6.0*sim.A_panel, 0.005)
    Q_int = st.number_input("Internal dissipation Q_internal (W)", 0.0, 50.0, 0.0, 0.1)

    T_c, Qs, Qa, Qir, Qin, Qtot = sim.thermal_equilibrium(A_abs=A_abs, A_rad=A_rad, Q_internal_W=Q_int)
    st.metric("Equilibrium temperature (°C)", f"{T_c:.2f}")

    dfQ = pd.DataFrame([{"Solar_avg_W": Qs, "Albedo_W": Qa, "Earth_IR_W": Qir,
                         "Internal_W": Qin, "Total_abs_W": Qtot}])
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
        "attitude": attitude, "auto_cal": auto_cal,
        "elec_derate": elec_derate, "beta_deg": beta_deg
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
    P_inst = P_inst * cal_factor * elec_derate

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
    "β-angle support with OAP vs β sweep; eclipse & panel incidence use Sun vector; "
    "correct Earth view factor; thermal with internal heat; aligned ECI→ECEF ground track; "
    "play buttons inside plots; electrical derate slider; and Pro-gated Save/Load/Export."
)
