# app.py — GeneSat-1–tuned CubeSat simulator (simple & complete)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Constants (SI)
# -------------------------
MU = 3.986004418e14       # Earth's gravitational parameter (m^3/s^2)
R_E = 6371e3              # Earth radius (m)
SIGMA = 5.670374419e-8    # Stefan–Boltzmann (W/m^2/K^4)
SOLAR_CONST = 1366.0      # Solar constant (W/m^2)
EARTH_IR = 237.0          # Approx Earth IR (W/m^2)
ALBEDO = 0.3              # Mean Earth albedo
OMEGA_E = 7.2921159e-5    # Earth rotation rate (rad/s)

# -------------------------
# GeneSat-1-ish defaults (you can tweak in sidebar)
# -------------------------
DEFAULTS = {
    "altitude_km": 460.0,     # GeneSat-1-like orbit
    "incl_deg": 40.0,
    "mass_kg": 4.6,
    "Cd": 2.2,
    "face_area_m2": 0.03,     # one flat panel (effective)
    "panel_eff": 0.25,        # electrical conversion efficiency
    "absorptivity": 0.65,
    "emissivity": 0.83,
    "target_avg_power_W": 4.5 # calibration target (approx GeneSat avg)
}

# -------------------------
# Atmosphere model (very simple MSIS-like)
# ρ(h) = ρ_ref * exp(-(h - h_ref)/H)
# Use ~60 km scale height in 300–600 km band, with ρ_ref at 200 km.
# -------------------------
def rho_msis_simple(h_m):
    H = 60e3
    rho_200 = 2.5e-11  # kg/m^3 at ~200 km (rough)
    h = np.maximum(h_m, 200e3)
    rho = rho_200 * np.exp(-(h - 200e3)/H)
    return np.clip(rho, 1e-13, None)  # floor to avoid zero

# -------------------------
# Utilities
# -------------------------
def clamp_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def eclipse_mask_xyz(x, y, z):
    """
    Simple cylindrical shadow test with Sun along +X.
    Eclipse when x < 0 (sat behind Earth wrt Sun) AND sqrt(y^2 + z^2) < R_E.
    Inputs in meters or kilometers consistently (we use km here in plotting).
    Returns boolean array (True = eclipsed).
    """
    r_perp = np.sqrt(y**2 + z**2)
    return (x < 0) & (r_perp < (R_E/1000.0))

def view_factor_earth(alt_m):
    # Solid-angle cap fraction: (1 - cos ψ)/2, ψ = arccos(R_E / r)
    r = R_E + alt_m
    ratio = np.clip(R_E / r, -1.0, 1.0)
    psi = np.arccos(ratio)
    return (1.0 - np.cos(psi)) / 2.0

# -------------------------
# CubeSat model
# -------------------------
class CubeSatSim:
    def __init__(self, altitude_km, incl_deg, mass_kg, Cd, face_area_m2, panel_eff, absorptivity, emissivity):
        self.h_km = float(altitude_km)
        self.h = self.h_km * 1000.0
        self.i = np.radians(float(incl_deg))
        self.m = float(mass_kg)
        self.Cd = float(Cd)
        self.A_face = float(face_area_m2)
        self.eta = float(panel_eff)
        self.alpha = float(absorptivity)
        self.eps = float(emissivity)

        self.r = R_E + self.h
        self.T_orbit = 2*np.pi*np.sqrt(self.r**3/MU)
        self.n = 2*np.pi/self.T_orbit
        self.v = np.sqrt(MU/self.r)  # circular orbit speed
        self.VF = view_factor_earth(self.h)

    def eclipse_fraction_geom(self):
        ratio = np.clip(R_E/(R_E + self.h), -1, 1)
        psi = np.arccos(ratio)
        return float(psi/np.pi)  # 0..0.5

    def orbit_xyz(self, N=720):
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        r_km = self.r/1000.0
        # simple inclined ring (Sun along +X)
        x = r_km*np.cos(theta)
        y = r_km*np.sin(theta)*np.cos(self.i)
        z = r_km*np.sin(theta)*np.sin(self.i)
        return theta, x, y, z

    def ground_track(self, N=720):
        t = np.linspace(0.0, self.T_orbit, N, endpoint=False)
        lon_inertial = self.n * t
        lon_earth = lon_inertial - OMEGA_E * t
        lon_deg = clamp_angle_deg(np.degrees(lon_earth))
        lat_deg = np.degrees(np.arcsin(np.sin(self.i) * np.sin(self.n*t)))
        return lon_deg, lat_deg

    # --- Attitude models & instantaneous incidence ---
    # Sun unit vector in ECI assumed +X. Panel normal depends on attitude.
    # - body-spin: average cosine -> constant 0.5 during sunlight
    # - sun-tracking: panel normal = +X always
    # - nadir-pointing: panel normal points to -r_hat; incidence = dot(-r_hat, +xhat)
    def panel_normal(self, theta, x, y, z, attitude):
        if attitude == "sun-tracking":
            # always toward Sun (+X)
            nx = np.ones_like(theta)
            ny = np.zeros_like(theta)
            nz = np.zeros_like(theta)
        elif attitude == "nadir-pointing":
            # -r_hat (pointing toward Earth center)
            rvec = np.vstack((x, y, z)).T
            rnorm = np.linalg.norm(rvec, axis=1, keepdims=True)
            nhat = -(rvec / np.maximum(rnorm, 1e-9))
            nx, ny, nz = nhat[:,0], nhat[:,1], nhat[:,2]
        else:
            # body-spin handled as average (we don't use instantaneous normal)
            nx = ny = nz = None
        return nx, ny, nz

    def instantaneous_power(self, theta, x, y, z, attitude, A_panel, eta, eclipsed_mask):
        # Sun direction +X
        sx, sy, sz = 1.0, 0.0, 0.0
        if attitude == "body-spin":
            # flat average incidence during sunlight
            cos_inc = np.full_like(theta, 0.5)
        else:
            nx, ny, nz = self.panel_normal(theta, x, y, z, attitude)
            # cos(incidence) = dot(nhat, shat)
            cos_inc = nx*sx + ny*sy + nz*sz
            cos_inc = np.clip(cos_inc, 0.0, 1.0)
        # zero during eclipse
        cos_inc = cos_inc * (~eclipsed_mask)
        P = SOLAR_CONST * A_panel * eta * cos_inc
        return P, cos_inc

    # Average generated power over an orbit, per attitude
    def avg_power(self, attitude, A_panel, eta):
        th, x, y, z = self.orbit_xyz(N=1440)
        eclip = eclipse_mask_xyz(x, y, z)
        if attitude == "body-spin":
            cos_mean = 0.5 * (~eclip)  # 0.5 when sunlit, 0 in eclipse
            P = SOLAR_CONST * A_panel * eta * cos_mean
        else:
            P, _ = self.instantaneous_power(th, x, y, z, attitude, A_panel, eta, eclip)
        return float(P.mean())

    # Thermal: radiative equilibrium using averaged loads
    def thermal_equilibrium(self, A_face=None, A_rad=None):
        if A_face is None:
            A_face = self.A_face
        if A_rad is None:
            A_rad = 6.0*self.A_face
        efrac = self.eclipse_fraction_geom()
        Q_solar = SOLAR_CONST * self.alpha * A_face * (1.0 - efrac)
        Q_alb   = ALBEDO * SOLAR_CONST * self.alpha * A_face * self.VF * (1.0 - efrac)
        Q_ir    = EARTH_IR * self.eps * A_face * self.VF
        Q_abs   = Q_solar + Q_alb + Q_ir
        T_k     = (Q_abs/(self.eps*SIGMA*A_rad))**0.25
        return float(T_k - 273.15), Q_solar, Q_alb, Q_ir, Q_abs

    # Drag decay using energy balance:
    # da/dt = - (Cd*A / m) * sqrt(μ a) * ρ(h)
    # integrate daily
    def drag_decay_days(self, days, A_ref=None):
        A = self.A_face if A_ref is None else A_ref
        a = R_E + self.h  # initial semi-major axis (circular)
        out_alt = []
        for _ in range(int(days)):
            rho = rho_msis_simple(a - R_E)
            da_dt = - (self.Cd * A / self.m) * np.sqrt(MU * a) * rho
            a = a + da_dt * 86400.0
            out_alt.append(a - R_E)
            if a < R_E + 120e3:  # burn-in safeguard
                a = R_E + 120e3
        return np.array(out_alt) / 1000.0  # km

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="GeneSat-1 Simple Simulator", layout="wide")
st.title("🛰️ GeneSat-1 Simple Simulator (with Play, shading, attitude, drag)")

with st.sidebar:
    st.header("Mission parameters (GeneSat-1-ish)")
    altitude_km = st.slider("Altitude (km)", 200.0, 600.0, DEFAULTS["altitude_km"], step=1.0)
    incl_deg    = st.slider("Inclination (deg)", 0.0, 98.0, DEFAULTS["incl_deg"], step=0.1)
    mass_kg     = st.number_input("Mass (kg)", 0.1, 50.0, DEFAULTS["mass_kg"], step=0.1)
    Cd          = st.slider("Drag coefficient Cd", 1.5, 3.0, DEFAULTS["Cd"], step=0.1)
    face_area   = st.number_input("Panel/face area (m²)", 0.001, 0.5, DEFAULTS["face_area_m2"], step=0.001)
    eta         = st.slider("Panel efficiency η", 0.05, 0.35, DEFAULTS["panel_eff"], step=0.01)
    alpha       = st.slider("Absorptivity α", 0.1, 1.0, DEFAULTS["absorptivity"], step=0.01)
    eps         = st.slider("Emissivity ε", 0.1, 1.0, DEFAULTS["emissivity"], step=0.01)
    attitude    = st.radio("Attitude model", ["body-spin", "sun-tracking", "nadir-pointing"])
    mission_days= st.slider("Mission duration (days)", 1, 180, 60)
    anim_speed  = st.slider("Animation speed (Plotly)", 0.1, 5.0, 1.0, 0.1)
    show_play   = st.checkbox("Show Play buttons on plots", True)
    do_cal      = st.checkbox("Calibrate to GeneSat target avg power (≈4.5 W)")

# Instantiate
sat = CubeSatSim(altitude_km, incl_deg, mass_kg, Cd, face_area, eta, alpha, eps)

# Optional 1): Calibration (simple multiplicative factor on η*A to hit target avg in current attitude)
cal_factor = 1.0
if do_cal:
    P_now = sat.avg_power(attitude, face_area, eta)
    target = DEFAULTS["target_avg_power_W"]
    if P_now > 0:
        cal_factor = target / P_now

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["3D + Shaded Orbit", "Power (per-orbit & avg)", "Thermal", "Drag Decay"])

# -------------------------
# Tab 1: 3D with sunlit/eclipse shading + Play
# -------------------------
with tab1:
    st.subheader("3D Orbit with Sunlit / Eclipse Shading")

    th, x_km, y_km, z_km = sat.orbit_xyz(N=720)
    mask_ecl = eclipse_mask_xyz(x_km, y_km, z_km)

    # Build a colored line: sunlit segments gold, eclipse gray
    # We'll draw as two traces for simplicity
    x_sun, y_sun, z_sun = x_km.copy(), y_km.copy(), z_km.copy()
    x_sun[mask_ecl] = None; y_sun[mask_ecl] = None; z_sun[mask_ecl] = None
    x_ecl, y_ecl, z_ecl = x_km.copy(), y_km.copy(), z_km.copy()
    x_ecl[~mask_ecl] = None; y_ecl[~mask_ecl] = None; z_ecl[~mask_ecl] = None

    fig3d = go.Figure()
    # Earth surface (semi-transparent)
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    xE = (R_E/1000.0)*np.cos(uu)*np.sin(vv)
    yE = (R_E/1000.0)*np.sin(uu)*np.sin(vv)
    zE = (R_E/1000.0)*np.cos(vv)
    fig3d.add_trace(go.Surface(x=xE, y=yE, z=zE, opacity=0.35, showscale=False, name="Earth"))

    fig3d.add_trace(go.Scatter3d(x=x_sun, y=y_sun, z=z_sun, mode="lines",
                                 line=dict(color="gold", width=4), name="Sunlit"))
    fig3d.add_trace(go.Scatter3d(x=x_ecl, y=y_ecl, z=z_ecl, mode="lines",
                                 line=dict(color="gray", width=4), name="Eclipse"))
    # satellite marker frames
    frames = []
    step = 4
    for k in range(0, len(x_km), step):
        frames.append(go.Frame(
            name=str(k),
            data=[go.Scatter3d(x=[x_km[k]], y=[y_km[k]], z=[z_km[k]],
                               mode="markers", marker=dict(size=6, color="red"))]
        ))
    fig3d.add_trace(go.Scatter3d(x=[x_km[0]], y=[y_km[0]], z=[z_km[0]],
                                 mode="markers", marker=dict(size=6, color="red"), name="Sat"))
    fig3d.frames = frames
    fig3d.update_layout(scene=dict(aspectmode="data"), height=550, showlegend=True,
                        margin=dict(l=0,r=0,t=0,b=0))
    if show_play:
        fig3d.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05, x=0.06,
                buttons=[dict(label="Play", method="animate",
                              args=[None, dict(frame=dict(duration=int(40/anim_speed), redraw=True),
                                               fromcurrent=True, mode="immediate")])]
            )]
        )
    st.plotly_chart(fig3d, use_container_width=True)

    col = st.columns(3)
    col[0].metric("Orbital period (min)", f"{sat.T_orbit/60.0:.2f}")
    col[1].metric("Eclipse fraction", f"{sat.eclipse_fraction_geom():.3f}")
    col[2].metric("Earth view factor", f"{sat.VF:.3f}")

# -------------------------
# Tab 2: Power — instantaneous & average; calibration option
# -------------------------
with tab2:
    st.subheader("Power vs Orbit Phase (with attitude)")

    Np = 720
    th, x_km, y_km, z_km = sat.orbit_xyz(N=Np)
    eclip = eclipse_mask_xyz(x_km, y_km, z_km)

    # Instantaneous power (before calibration factor)
    P_inst, cos_inc = sat.instantaneous_power(
        th, x_km, y_km, z_km, attitude, A_panel=face_area, eta=eta, eclipsed_mask=eclip
    ) if attitude != "body-spin" else (
        SOLAR_CONST*face_area*eta*(~eclip)*0.5, np.full(Np, 0.5)*(~eclip)
    )

    # Apply simple calibration scaling to match target average power (if selected)
    P_inst = P_inst * cal_factor

    # Per-orbit average and daily energy
    P_avg = float(P_inst.mean())
    day_Wh = P_avg * 86400.0 / 3600.0

    # Consumption and SoC
    cons_W = st.slider("Average consumption (W)", 0.1, 15.0, 3.0, 0.1)
    batt_Wh = st.slider("Battery capacity (Wh)", 5.0, 200.0, 30.0, 1.0)
    start_soc = st.slider("Start SoC (%)", 0, 100, 50)

    daily_cons_Wh = cons_W * 24.0
    net_daily = day_Wh - daily_cons_Wh

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
    c2.metric("Daily generation (Wh)", f"{day_Wh:.1f}")
    c3.metric("Daily consumption (Wh)", f"{daily_cons_Wh:.1f}")

    # Plots
    figP = px.line(x=np.degrees(th), y=P_inst, labels={"x":"Orbit phase (deg)", "y":"Power (W)"},
                   title=f"Instantaneous Power — {attitude} (cal factor {cal_factor:.2f})")
    st.plotly_chart(figP, use_container_width=True)
    st.plotly_chart(px.line(df_soc, x="Day", y="SoC (%)", markers=True,
                            title="Battery SoC over mission"), use_container_width=True)

    if df_soc["SoC (%)"].iloc[-1] < 20:
        st.warning("Projected final SoC < 20% — consider more area/efficiency, better pointing, or less load.")
    else:
        st.success("Battery SoC projection looks acceptable.")

# -------------------------
# Tab 3: Thermal
# -------------------------
with tab3:
    st.subheader("Radiative Thermal Balance (Averaged)")
    A_face = st.number_input("Absorbing face area (m²)", 0.001, 0.5, face_area, 0.001)
    A_rad  = st.number_input("Radiating area (m²)", 0.01, 3.0, 6.0*face_area, 0.01)

    T_c, Qs, Qa, Qir, Qabs = sat.thermal_equilibrium(A_face=A_face, A_rad=A_rad)
    st.metric("Equilibrium temperature (°C)", f"{T_c:.2f}")

    dfQ = pd.DataFrame([{"Solar_avg_W": Qs, "Albedo_W": Qa, "Earth_IR_W": Qir, "Total_abs_W": Qabs}])
    st.dataframe(dfQ, use_container_width=True)
    st.plotly_chart(px.bar(dfQ.melt(var_name="Component", value_name="W"),
                           x="Component", y="W", title="Absorbed power components"),
                    use_container_width=True)

# -------------------------
# Tab 4: Drag Decay (simple MSIS-like)
# -------------------------
with tab4:
    st.subheader("Altitude Decay from Drag (simple)")
    A_drag = st.number_input("Reference drag area (m²)", 0.001, 0.5, face_area, 0.001)
    alt_km_series = sat.drag_decay_days(mission_days, A_ref=A_drag)
    df_alt = pd.DataFrame({"Day": np.arange(1, mission_days+1), "Altitude (km)": alt_km_series})
    st.plotly_chart(px.line(df_alt, x="Day", y="Altitude (km)", markers=True,
                            title="Altitude decay over mission"),
                    use_container_width=True)

st.markdown("---")
st.caption("Includes: Play buttons, 3D sunlit/eclipsed shading, attitude incidence, simple GeneSat-1 power calibration, and MSIS-like drag decay.")
