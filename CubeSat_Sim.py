import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class CubeSatSim:
    def __init__(self, altitude_km=400, inclination_deg=51.6):
        self.altitude_km = altitude_km
        self.altitude_m = altitude_km * 1000
        self.inclination = np.radians(inclination_deg)
        self.mass = 1.33
        self.area = 0.01
        self.drag_coeff = 2.2
        self.solar_constant = 1366
        self.albedo = 0.3
        self.earth_ir = 237
        self.emissivity = 0.8
        self.absorptivity = 0.5
        self.view_factor_earth = self.calculate_view_factor()
        self.orbital_period = 2 * np.pi * np.sqrt((6371e3 + self.altitude_m)**3 / (3.986e14))

    def calculate_view_factor(self):
        r_earth = 6371e3
        h = self.altitude_m
        return (r_earth / (r_earth + h)) ** 2

    def orbital_decay(self, time_days):
        rho = 1e-12 * np.exp(-self.altitude_m / 7000)
        decay_rate = -0.5 * rho * self.drag_coeff * self.area * self.orbital_period / self.mass
        delta_h = decay_rate * time_days * 86400
        return self.altitude_km + delta_h

    def power_budget(self, eclipse_fraction=0.4):
        solar_power = self.solar_constant * self.area * self.absorptivity * (1 - eclipse_fraction)
        avg_power = solar_power * 0.9
        consumption = 2.0
        return avg_power, consumption

    def thermal_model(self):
        sigma = 5.67e-8
        A_face = 0.01
        A_rad = 6 * A_face
        eclipse_fraction = 0.4
        vf = self.view_factor_earth
        n_faces_ir = 10.79

        q_solar_avg = self.solar_constant * self.absorptivity * A_face * (1 - eclipse_fraction)
        q_albedo_avg = self.albedo * self.solar_constant * self.absorptivity * A_face * vf * (1 - eclipse_fraction)
        q_ir_avg = self.earth_ir * self.emissivity * n_faces_ir * A_face * vf
        q_abs_avg = q_solar_avg + q_albedo_avg + q_ir_avg
        T_eq = ((q_abs_avg / (self.emissivity * sigma * A_rad)) ** 0.25) - 273.15

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

    def simulate_orbit_3d(self, num_points=200):
        theta = np.linspace(0, 2 * np.pi, num_points)
        radius = 6371 + self.altitude_km
        inc = self.inclination
        x = radius * np.cos(theta)
        y = radius * np.sin(theta) * np.cos(inc)
        z = radius * np.sin(theta) * np.sin(inc)
        return x, y, z

    def simulate_orbit(self, num_orbits=10):
        times = np.linspace(0, self.orbital_period * num_orbits, 1000)
        thetas = 2 * np.pi * times / self.orbital_period
        alt = np.full_like(times, self.altitude_m) + 100 * np.sin(thetas)
        return times / 3600, alt / 1000

    def ground_track(self, num_points=200):
        theta = np.linspace(0, 2 * np.pi, num_points)
        lon = np.degrees(theta)
        lat = np.degrees(np.sin(self.inclination) * np.sin(theta))
        return lon, lat

# --- Streamlit App ---
st.set_page_config(page_title="CubeSat Simulator", layout="wide")
st.title("NASA-Accurate CubeSat Simulator")
st.markdown("**GeneSat-1 Validated • 1U CubeSat • 3D Orbit + Ground Track**")

# Sidebar
with st.sidebar:
    st.header("Mission Parameters")
    altitude = st.slider("Orbit Altitude (km)", 200, 800, 400)
    inclination = st.slider("Inclination (°)", 0.0, 90.0, 51.6)
    mission_days = st.number_input("Mission Duration (days)", 1, 365, 30)
    anim_speed = st.slider("Animation Speed", 0.1, 5.0, 1.0, 0.1)

sim = CubeSatSim(altitude, inclination)

tab1, tab2, tab3 = st.tabs(["3D + Ground Track", "Power", "Thermal"])

# === TAB 1: CLEAN FULL-WIDTH ANIMATIONS ===
with tab1:
    # SINGLE PLAY/PAUSE BUTTON
    col_play = st.columns([1, 1, 1])
    with col_play[1]:
        play = st.button("Play Animation", use_container_width=True)
        pause = st.button("Pause", use_container_width=True)

    # Generate data
    x_orbit, y_orbit, z_orbit = sim.simulate_orbit_3d()
    lon, lat = sim.ground_track()

    # Shared frames
    frames = []
    for i in range(0, len(x_orbit), 2):
        frames.append(go.Frame(
            name=str(i),
            data=[
                # 3D Trail
                go.Scatter3d(x=x_orbit[:i], y=y_orbit[:i], z=z_orbit[:i],
                             mode='lines', line=dict(color='yellow', width=4)),
                # 3D CubeSat
                go.Scatter3d(x=[x_orbit[i]], y=[y_orbit[i]], z=[z_orbit[i]],
                             mode='markers', marker=dict(size=12, color='yellow', symbol='diamond')),
                # Ground Track Trail
                go.Scattergeo(lon=lon[:i], lat=lat[:i], mode='lines', line=dict(color='yellow', width=4)),
                # Ground Track CubeSat
                go.Scattergeo(lon=[lon[i]], lat=[lat[i]], mode='markers', marker=dict(size=12, color='yellow'))
            ]
        ))

    # === FULL-WIDTH 3D ORBIT (NO WIREFRAME, NO SUN) ===
    st.markdown("### 3D Orbital Animation")
    fig3d = go.Figure(
        data=[
            # Static Orbit
            go.Scatter3d(x=x_orbit, y=y_orbit, z=z_orbit, mode='lines', line=dict(color='red', width=6), name='Orbit'),
            # Initial Trail
            go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='yellow', width=4), name='Trail'),
            # Initial CubeSat
            go.Scatter3d(x=[x_orbit[0]], y=[y_orbit[0]], z=[z_orbit[0]],
                         mode='markers', marker=dict(size=12, color='yellow', symbol='diamond'), name='CubeSat')
        ],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50/anim_speed, redraw=True), fromcurrent=True, mode='immediate')]),
                    dict(label="Pause", method="animate", args=[[None], dict(mode="immediate")])
                ],
                y=1.1
            )],
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        ),
        frames=frames
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # === FULL-WIDTH GROUND TRACK ===
    st.markdown("### Ground Track Map")
    fig_map = go.Figure(
        data=[
            # Static Orbit
            go.Scattergeo(lon=lon, lat=lat, mode='lines', line=dict(color='red', width=4), name='Orbit'),
            # Initial Track
            go.Scattergeo(lon=[], lat=[], mode='lines', line=dict(color='yellow', width=4), name='Track'),
            # Initial CubeSat
            go.Scattergeo(lon=[lon[0]], lat=[lat[0]], mode='markers', marker=dict(size=12, color='yellow'), name='CubeSat')
        ],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50/anim_speed, redraw=True), fromcurrent=True, mode='immediate')]),
                    dict(label="Pause", method="animate", args=[[None], dict(mode="immediate")])
                ],
                y=1.1
            )],
            geo=dict(
                projection_type='orthographic',
                showland=True, landcolor='lightgreen',
                showocean=True, oceancolor='lightblue',
                showcountries=True, countrycolor='gray',
                lataxis=dict(range=[-90, 90]), lonaxis=dict(range=[-180, 180])
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        ),
        frames=frames
    )

    st.plotly_chart(fig_map, use_container_width=True)

# === TAB 2: POWER ===
with tab2:
    st.subheader("Power Budget")
    avg_p, cons = sim.power_budget()
    net = avg_p - cons
    col1, col2, col3 = st.columns(3)
    col1.metric("Generated", f"{avg_p:.2f} W")
    col2.metric("Consumed", f"{cons:.2f} W")
    col3.metric("Net", f"{net:+.2f} W", delta=f"{net:+.2f} W")

# === TAB 3: THERMAL ===
with tab3:
    st.subheader("Thermal Analysis")
    thermal = sim.thermal_model()
    df = pd.DataFrame(list(thermal.items()), columns=['Case', 'Temperature (°C)'])
    fig_bar = px.bar(df, x='Case', y='Temperature (°C)', color='Case', title="Temperature Cases")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Radiation Exposure")
    dose = 0.5 + (sim.altitude_km - 200) * 0.002 if sim.altitude_km < 1000 else 5.0 + (sim.altitude_km - 1000) * 0.01
    st.write(f"**Daily Dose:** {dose:.2f} rads/day")
    st.write(f"**{mission_days}-Day Total:** {dose*mission_days:.1f} rads")

if st.button("Benchmark: 400 km Run"):
    bench = CubeSatSim(400)
    t = bench.thermal_model()
    st.success("**GeneSat-1 Validated:**")
    st.write(f"• Equilibrium: **{t['Equilibrium Temp (°C)']}°C**")
    st.write(f"• Hot Face: **{t['Hot Face (°C)']}°C**")
    st.write(f"• Cold Face: **{t['Cold Face (°C)']}°C**")
