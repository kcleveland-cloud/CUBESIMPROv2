import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
#                   CubeSat Simulation Class
# ==========================================================

class CubeSatSim:
    def __init__(self, altitude_km=500, inclination_deg=51.6, area_m2=0.1, absorptivity=0.6, emissivity=0.8):
        self.altitude = altitude_km
        self.inclination = np.radians(inclination_deg)
        self.area = area_m2
        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.earth_radius = 6371  # km
        self.mu = 398600  # km^3/s^2
        self.orbital_period = 2 * np.pi * np.sqrt(((self.earth_radius + self.altitude) ** 3) / self.mu)
        self.solar_constant = 1361  # W/m^2

    # ---- Orbital Simulation ----
    def simulate_orbit_3d(self, num_points=500):
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = self.earth_radius + self.altitude
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(self.inclination)
        z = r * np.sin(theta) * np.sin(self.inclination)
        return x, y, z

    # ---- Ground Track Simulation ----
    def ground_track(self, num_points=500):
        t = np.linspace(0, self.orbital_period, num_points)
        mean_motion = 2 * np.pi / self.orbital_period
        lon = np.degrees(mean_motion * t) % 360 - 180
        lat = np.degrees(np.sin(mean_motion * t) * np.sin(self.inclination))
        return lon, lat

    # ---- Power Budget ----
    def power_budget(self, eclipse_fraction=None):
        if eclipse_fraction is None:
            eclipse_fraction = 1 - (np.cos(self.inclination) ** 0.5)
        solar_power = self.solar_constant * self.area * self.absorptivity * (1 - eclipse_fraction)
        return solar_power * 0.85, 2.0  # (generated, consumed)

    # ---- Thermal Model ----
    def thermal_model(self, sunlit_fraction=0.6):
        q_in = self.solar_constant * self.absorptivity * sunlit_fraction
        q_out = 5.67e-8 * self.emissivity * (300 ** 4)
        equilibrium_temp = ((q_in + q_out) / (5.67e-8 * self.emissivity)) ** 0.25
        return equilibrium_temp - 273.15  # °C


# ==========================================================
#                   Streamlit Interface
# ==========================================================

st.set_page_config(page_title="🛰️ CubeSat Mission Simulator", layout="wide")

st.title("🛰️ CubeSat Mission Simulator")
st.markdown("Visualize and analyze CubeSat orbit, power, and thermal behavior interactively.")

# --- Sidebar Controls ---
st.sidebar.header("Mission Parameters")
altitude = st.sidebar.slider("Altitude (km)", 200, 2000, 500, step=50)
inclination = st.sidebar.slider("Inclination (deg)", 0, 98, 51, step=1)
area = st.sidebar.slider("Solar Panel Area (m²)", 0.01, 0.5, 0.1, step=0.01)
absorptivity = st.sidebar.slider("Absorptivity", 0.1, 1.0, 0.6, step=0.05)
emissivity = st.sidebar.slider("Emissivity", 0.1, 1.0, 0.8, step=0.05)

sim = CubeSatSim(altitude, inclination, area, absorptivity, emissivity)


# --- Cache heavy computations ---
@st.cache_data
def get_orbit_data(altitude, inclination):
    sim = CubeSatSim(altitude, inclination)
    orbit = sim.simulate_orbit_3d()
    track = sim.ground_track()
    return orbit, track


orbit_data, track_data = get_orbit_data(altitude, inclination)
x_orbit, y_orbit, z_orbit = orbit_data
lon, lat = track_data
earth_radius = 6371

# ==========================================================
#                           Tabs
# ==========================================================

tab1, tab2, tab3 = st.tabs(["🌍 Orbit Visualization", "🔋 Power Budget", "🌡️ Thermal Analysis"])

# ==========================================================
#                     ORBIT VISUALIZATION
# ==========================================================

with tab1:
    st.subheader("3D Orbit and Ground Tracks")

    col1, col2, col3 = st.columns(3)
    play_wire = col1.toggle("🎞️ Animate Wire Earth", False)
    play_flat = col2.toggle("🎞️ Animate Flat Earth", False)
    play_globe = col3.toggle("🎞️ Animate Globe Earth", False)
    anim_speed = st.slider("Animation Speed", 0.1, 5.0, 1.0)

    # --------- 3D ORBIT (Wire Earth) -----------
    fig_wire = go.Figure()

    # Earth surface (semi-transparent)
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_earth = earth_radius * np.cos(u) * np.sin(v)
    y_earth = earth_radius * np.sin(u) * np.sin(v)
    z_earth = earth_radius * np.cos(v)

    fig_wire.add_surface(x=x_earth, y=y_earth, z=z_earth,
                         colorscale="Blues", opacity=0.4, showscale=False)

    fig_wire.add_trace(go.Scatter3d(x=x_orbit, y=y_orbit, z=z_orbit,
                                    mode="lines", line=dict(color="yellow", width=4),
                                    name="Orbit"))

    frames_wire = [
        go.Frame(name=str(i),
                 data=[go.Scatter3d(
                     x=x_orbit[:i], y=y_orbit[:i], z=z_orbit[:i],
                     mode="lines", line=dict(color="yellow", width=4)),
                       go.Scatter3d(x=[x_orbit[i]], y=[y_orbit[i]], z=[z_orbit[i]],
                                    mode="markers", marker=dict(size=10, color="red"))])
        for i in range(0, len(x_orbit), 2)
    ]

    fig_wire.frames = frames_wire
    fig_wire.update_layout(scene=dict(aspectmode="data"),
                           margin=dict(l=0, r=0, b=0, t=0),
                           showlegend=False)

    if play_wire:
        fig_wire.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.1,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=int(50 / anim_speed), redraw=True),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                )]
            )]
        )

    st.plotly_chart(fig_wire, use_container_width=True)

    # --------- FLAT EARTH -----------
    fig_flat = px.scatter_geo(lat=lat, lon=lon, title="Ground Track (Flat Map)")
    fig_flat.update_geos(projection_type="equirectangular", showcoastlines=True, showcountries=True)

    frames_flat = [
        go.Frame(name=str(i),
                 data=[go.Scattergeo(lat=lat[:i], lon=lon[:i], mode="lines",
                                     line=dict(color="yellow", width=2)),
                       go.Scattergeo(lat=[lat[i]], lon=[lon[i]],
                                     mode="markers", marker=dict(size=8, color="red"))])
        for i in range(0, len(lon), 2)
    ]

    fig_flat.frames = frames_flat
    if play_flat:
        fig_flat.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.12,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=int(50 / anim_speed), redraw=True),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                )]
            )]
        )

    st.plotly_chart(fig_flat, use_container_width=True)

    # --------- GLOBE VIEW -----------
    fig_globe = px.scatter_geo(lat=lat, lon=lon, projection="orthographic", title="Ground Track (Globe)")
    frames_globe = [
        go.Frame(name=str(i),
                 data=[go.Scattergeo(lat=lat[:i], lon=lon[:i], mode="lines",
                                     line=dict(color="yellow", width=2)),
                       go.Scattergeo(lat=[lat[i]], lon=[lon[i]],
                                     mode="markers", marker=dict(size=8, color="red"))])
        for i in range(0, len(lon), 2)
    ]

    fig_globe.frames = frames_globe
    if play_globe:
        fig_globe.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.14,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=int(50 / anim_speed), redraw=True),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                )]
            )]
        )

    st.plotly_chart(fig_globe, use_container_width=True)


# ==========================================================
#                     POWER BUDGET
# ==========================================================

with tab2:
    st.subheader("Power Budget")
    generated, consumed = sim.power_budget()
    efficiency = generated / consumed * 100 if consumed > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Generated Power (W)", f"{generated:.2f}")
    col2.metric("Consumed Power (W)", f"{consumed:.2f}")
    col3.metric("Efficiency (%)", f"{efficiency:.1f}")

    labels = ["Generated", "Consumed"]
    values = [generated, consumed]
    st.plotly_chart(px.pie(values=values, names=labels, title="Power Budget Breakdown"), use_container_width=True)


# ==========================================================
#                     THERMAL ANALYSIS
# ==========================================================

with tab3:
    st.subheader("Thermal Analysis")
    sunlit_fraction = st.slider("Sunlit Fraction", 0.0, 1.0, 0.6, step=0.05)
    temperature = sim.thermal_model(sunlit_fraction)

    st.metric("Equilibrium Temperature (°C)", f"{temperature:.1f}")
    st.plotly_chart(
        px.line(x=[0, 1], y=[temperature - 10, temperature + 10],
                labels={'x': "Time (normalized)", 'y': "Temperature (°C)"},
                title="Temperature Variation (Simplified)"),
        use_container_width=True
    )

# ==========================================================
#                     FOOTER
# ==========================================================

st.markdown("---")
st.caption("CubeSat Mission Simulator · Built with ❤️ using Streamlit & Plotly · 2025")
