import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# --- SAFE SESSION STATE ---
# ---------------------------
if 'play_wire' not in st.session_state:
    st.session_state.play_wire = False
if 'play_flat' not in st.session_state:
    st.session_state.play_flat = False
if 'play_globe' not in st.session_state:
    st.session_state.play_globe = False

# ---------------------------
# --- CubeSat Simulation ---
# ---------------------------
class CubeSatSim:
    def __init__(self, altitude_km=400, inclination_deg=51.6):
        self.altitude_km = altitude_km
        self.altitude_m = altitude_km * 1000.0
        self.inclination = np.radians(inclination_deg)
        self.mass = 1.33  # kg (1U-ish)
        self.area = 0.01  # m^2 (face area)
        self.drag_coeff = 2.2
        self.solar_constant = 1366  # W/m^2
        self.albedo = 0.3
        self.earth_ir = 237  # W/m^2
        self.emissivity = 0.8
        self.absorptivity = 0.5
        self.view_factor_earth = self.calculate_view_factor()
        # orbital period (s)
        self.orbital_period = 2 * np.pi * np.sqrt(((6371e3 + self.altitude_m) ** 3) / (3.986e14))

    def calculate_view_factor(self):
        r_earth = 6371e3
        h = self.altitude_m
        return (r_earth / (r_earth + h)) ** 2

    def power_budget(self, eclipse_fraction=None):
        # If not provided, estimate eclipse fraction from altitude using simple empirical relation
        if eclipse_fraction is None:
            # Larger altitude -> smaller view factor -> shorter eclipse fraction; crude mapping
            vf = self.view_factor_earth
            eclipse_fraction = min(max(0.2 + (1.0 - vf) * 0.4, 0.05), 0.6)

        # Instantaneous solar power on one face, average over orbit with eclipse
        solar_power_inst = self.solar_constant * self.area * self.absorptivity
        avg_power_gen = solar_power_inst * (1 - eclipse_fraction) * 0.85  # inefficiencies
        # Nominal payload + bus consumption (W) — simple baseline that could be parameterized
        baseline_consumption = 2.0  # W
        # Add small housekeeping margin
        consumption = baseline_consumption
        return avg_power_gen, consumption, eclipse_fraction

    def thermal_model(self):
        sigma = 5.67e-8
        A_face = self.area
        A_rad = 6 * A_face
        eclipse_fraction = 0.4
        vf = self.view_factor_earth
        n_faces_ir = 10.79

        q_solar_avg = self.solar_constant * self.absorptivity * A_face * (1 - eclipse_fraction)
        q_albedo_avg = self.albedo * self.solar_constant * self.absorptivity * A_face * vf * (1 - eclipse_fraction)
        q_ir_avg = self.earth_ir * self.emissivity * n_faces_ir * A_face * vf
        q_abs_avg = q_solar_avg + q_albedo_avg + q_ir_avg
        T_eq = ((q_abs_avg / (self.emissivity * sigma * A_rad)) ** 0.25) - 273.15

        # Hot and cold face approximations (still crude but consistent)
        q_hot = (
            self.solar_constant * self.absorptivity * A_face +
            self.albedo * self.solar_constant * self.absorptivity * A_face * vf +
            self.earth_ir * self.emissivity * A_face * vf
        )
        A_rad_hot = 0.00851
        T_hot = (q_hot / (self.emissivity * sigma * A_rad_hot)) ** 0.25 - 273.15

        q_cold = self.earth_ir * self.emissivity * A_face * vf
        A_rad_cold = 0.00933
        T_cold = (q_cold / (self.emissivity * sigma * A_rad_cold)) ** 0.25 - 273.15

        return {
            'Equilibrium Temp (°C)': round(T_eq, 1),
            'Hot Face (°C)': round(T_hot, 1),
            'Cold Face (°C)': round(T_cold, 1)
        }

    def simulate_orbit_3d(self, num_points=400):
        # Returns arrays in kilometers (for nicer numbers in plot)
        theta = np.linspace(0, 2 * np.pi, num_points)
        radius_km = 6371.0 + self.altitude_km
        inc = self.inclination
        x = radius_km * np.cos(theta)
        y = radius_km * np.sin(theta) * np.cos(inc)
        z = radius_km * np.sin(theta) * np.sin(inc)
        return x, y, z

    def ground_track(self, num_points=400):
        theta = np.linspace(0, 2 * np.pi, num_points)
        lon = (np.degrees(theta) + 180) % 360 - 180  # map to [-180,180]
        lat = np.degrees(np.sin(self.inclination) * np.sin(theta))
        return lon, lat


# ---------------------------
# --- Cached helpers ----
# ---------------------------
@st.cache_data
def get_orbit_data(altitude_km, inclination_deg, num_points=400):
    sim = CubeSatSim(altitude_km, inclination_deg)
    x, y, z = sim.simulate_orbit_3d(num_points=num_points)
    lon, lat = sim.ground_track(num_points=num_points)
    return sim, x, y, z, lon, lat


# ---------------------------
# --- Streamlit UI ----
# ---------------------------
st.set_page_config(page_title="CubeSat Simulator (Refactor)", layout="wide")
st.title("CubeSat Simulator — Refactored")
st.markdown("**Improvements:** separate frames, cached orbit, animation toggles, 3D surface Earth, battery SoC demo")

# Sidebar
with st.sidebar:
    st.header("Mission Parameters")
    altitude = st.slider("Orbit Altitude (km)", 200, 800, 400)
    inclination = st.slider("Inclination (°)", 0.0, 90.0, 51.6)
    mission_days = st.number_input("Mission Duration (days)", min_value=1, max_value=365, value=30)
    anim_speed = st.slider("Animation Speed (1 = baseline)", 0.1, 5.0, 1.0, 0.1)
    show_surface_earth = st.checkbox("Show surface Earth in 3D", value=True)
    run_battery_sim = st.checkbox("Run battery SoC simulation", value=True)

sim, x_orbit, y_orbit, z_orbit, lon, lat = get_orbit_data(altitude, inclination)

tab1, tab2, tab3 = st.tabs(["3D + Ground Track", "Power", "Thermal"])

# ---------------------------
# --- TAB 1: 3D + Ground Track
# ---------------------------
with tab1:
    st.subheader("Orbit Visualizations")

    # Animation control toggles (use checkbox instead of button + rerun)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        play_wire = st.checkbox("Play Wire Earth (3D orbit)", value=st.session_state.play_wire)
    with col_b:
        play_flat = st.checkbox("Play Flat Ground Track", value=st.session_state.play_flat)
    with col_c:
        play_globe = st.checkbox("Play Globe Ground Track", value=st.session_state.play_globe)

    # keep session state consistent
    st.session_state.play_wire = play_wire
    st.session_state.play_flat = play_flat
    st.session_state.play_globe = play_globe

    # Create frames separately to keep each figure light
    # 3D orbit frames (Scatter3d)
    frames_3d = []
    steps = 4
    for i in range(0, len(x_orbit), steps):
        frames_3d.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(x=x_orbit[:i], y=y_orbit[:i], z=z_orbit[:i],
                             mode='lines', line=dict(color='yellow', width=4)),
                go.Scatter3d(x=[x_orbit[i % len(x_orbit)]],
                             y=[y_orbit[i % len(y_orbit)]],
                             z=[z_orbit[i % len(z_orbit)]],
                             mode='markers',
                             marker=dict(size=6, color='red', symbol='diamond'))
            ]
        ))

    # 3D figure (wire orbit + optional Earth surface)
    fig_wire = go.Figure(
        data=[
            go.Scatter3d(x=x_orbit, y=y_orbit, z=z_orbit, mode='lines', line=dict(color='red', width=3)),
            go.Scatter3d(x=[x_orbit[0]], y=[y_orbit[0]], z=[z_orbit[0]],
                         mode='markers', marker=dict(size=8, color='yellow', symbol='diamond'))
        ],
        frames=frames_3d,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            height=550
        )
    )

    # Optional Earth surface (smoother than sparse wireframe)
    if show_surface_earth:
        # create a surface grid (kilometers)
        earth_radius = 6371.0
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        uu, vv = np.meshgrid(u, v)
        x_surf = earth_radius * np.cos(uu) * np.sin(vv)
        y_surf = earth_radius * np.sin(uu) * np.sin(vv)
        z_surf = earth_radius * np.cos(vv)
        fig_wire.add_trace(go.Surface(x=x_surf, y=y_surf, z=z_surf,
                                      opacity=0.35, showscale=False, hoverinfo='skip'))

    if play_wire:
        fig_wire.update_layout(updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.05,
            x=0.1,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True),
                                           fromcurrent=True, mode="immediate")])]))]

    st.plotly_chart(fig_wire, use_container_width=True)

    # Flat ground track frames (Scattergeo)
    frames_flat = []
    for i in range(0, len(lon), steps):
        frames_flat.append(go.Frame(
            name=str(i),
            data=[
                go.Scattergeo(lon=lon[:i], lat=lat[:i], mode='lines', line=dict(color='yellow', width=2)),
                go.Scattergeo(lon=[lon[i % len(lon)]], lat=[lat[i % len(lat)]],
                              mode='markers', marker=dict(size=6, color='red'))
            ]
        ))

    fig_flat = go.Figure(
        data=[
            go.Scattergeo(lon=lon, lat=lat, mode='lines', line=dict(color='red', width=2)),
            go.Scattergeo(lon=[lon[0]], lat=[lat[0]], mode='markers', marker=dict(size=6, color='yellow'))
        ],
        frames=frames_flat,
        layout=go.Layout(
            geo=dict(projection_type='natural earth', showland=True, landcolor='lightgreen',
                     showocean=True, oceancolor='lightblue'),
            height=400
        )
    )

    if play_flat:
        fig_flat.update_layout(updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.05,
            x=0.12,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True), fromcurrent=True, mode="immediate")])]))]

    st.plotly_chart(fig_flat, use_container_width=True)

    # Globe ground track frames (orthographic)
    frames_globe = []
    for i in range(0, len(lon), steps):
        frames_globe.append(go.Frame(
            name=str(i),
            data=[
                go.Scattergeo(lon=lon[:i], lat=lat[:i], mode='lines', line=dict(color='yellow', width=2)),
                go.Scattergeo(lon=[lon[i % len(lon)]], lat=[lat[i % len(lat)]],
                              mode='markers', marker=dict(size=6, color='red'))
            ]
        ))

    fig_globe = go.Figure(
        data=[
            go.Scattergeo(lon=lon, lat=lat, mode='lines', line=dict(color='red', width=2)),
            go.Scattergeo(lon=[lon[0]], lat=[lat[0]], mode='markers', marker=dict(size=6, color='yellow'))
        ],
        frames=frames_globe,
        layout=go.Layout(
            geo=dict(projection_type='orthographic', showland=True, landcolor='lightgreen',
                     showocean=True, oceancolor='lightblue'),
            height=400
        )
    )

    if play_globe:
        fig_globe.update_layout(updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.05,
            x=0.12,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=int(50/anim_speed), redraw=True), fromcurrent=True, mode="immediate")])]))]

    st.plotly_chart(fig_globe, use_container_width=True)


# ---------------------------
# --- TAB 2: Power ----
# ---------------------------
with tab2:
    st.subheader("Power Budget & Battery Simulation")
    avg_p, cons, eclipse_fraction = sim.power_budget()
    net = avg_p - cons

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Generation (W)", f"{avg_p:.3f}")
    c2.metric("Consumption (W)", f"{cons:.3f}")
    c3.metric("Net (W)", f"{net:+.3f}", delta=f"{net:+.3f}")

    st.markdown(f"Estimated eclipse fraction: **{eclipse_fraction:.2f}**")

    # Simple battery simulation over mission days
    if run_battery_sim:
        battery_capacity_wh = st.number_input("Battery capacity (Wh)", min_value=1.0, value=20.0)
        # assume orbit repeats many times per day; compute daily energy
        seconds_per_day = 86400
        orbits_per_day = seconds_per_day / sim.orbital_period
        daily_gen_wh = avg_p * seconds_per_day / 3600.0
        daily_cons_wh = cons * seconds_per_day / 3600.0
        days = np.arange(1, mission_days + 1)
        soc = []
        soc_val = 0.5 * battery_capacity_wh  # start at 50% SoC
        for d in days:
            soc_val += (daily_gen_wh - daily_cons_wh)
            # clamp
            soc_val = min(max(soc_val, 0.0), battery_capacity_wh)
            soc.append(100.0 * soc_val / battery_capacity_wh)

        df_soc = pd.DataFrame({'Day': days, 'SoC (%)': soc})
        # interactive table + chart
        st.write("Battery State of Charge over mission (start 50% SoC):")
        st.dataframe(df_soc)
        fig_soc = px.line(df_soc, x='Day', y='SoC (%)', markers=True)
        st.plotly_chart(fig_soc, use_container_width=True)

        final = soc[-1]
        if final < 20:
            st.warning("Projected end-of-mission SoC below 20% — consider larger battery or lower consumption.")
        else:
            st.success("Battery SoC looks acceptable for the chosen parameters.")


# ---------------------------
# --- TAB 3: Thermal ----
# ---------------------------
with tab3:
    st.subheader("Thermal Analysis")
    thermal = sim.thermal_model()
    df = pd.DataFrame(list(thermal.items()), columns=['Case', 'Temperature (°C)'])
    fig_bar = px.bar(df, x='Case', y='Temperature (°C)', color='Case')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Show numeric results
    st.table(df.set_index('Case'))

# ---------------------------
# --- Footer / Notes ----
# ---------------------------
st.markdown("---")
st.caption("Refactor: separate frames, caching, and non-destructive animation toggles. "
           "This is still a simplified physics model for demonstration and visualization.")
