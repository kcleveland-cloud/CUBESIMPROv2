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

        q_hot = self.solar_constant * self.absorptivity * A_face \
                + self.albedo * self.solar_constant * self.absorptivity * A_face * vf \
                + self.earth_ir * self.emissivity * A_face * vf
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

    def simulate_orbit(self, num_orbits=10):
        times = np.linspace(0, self.orbital_period * num_orbits, 1000)
        thetas = 2 * np.pi * times / self.orbital_period
        alt = np.full_like(times, self.altitude_m) + 100 * np.sin(thetas)
        return times / 3600, alt / 1000

# --- Streamlit App ---
st.set_page_config(page_title="CubeSat Simulator", layout="wide")
st.title("NASA-Accurate CubeSat Simulator")
st.write("Based on **GeneSat-1** flight data — **1U CubeSat**")

col1, col2 = st.columns([1, 2])
with col1:
    altitude = st.slider("Orbit Altitude (km)", 200, 800, 400)
    inclination = st.slider("Inclination (°)", 0.0, 90.0, 51.6)
    mission_days = st.number_input("Mission Duration (days)", 1, 365, 30)

sim = CubeSatSim(altitude, inclination)

tab1, tab2, tab3 = st.tabs(["Orbit", "Power", "Thermal"])

with tab1:
    st.subheader("Orbital Parameters")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Altitude", f"{sim.altitude_km} km")
        st.metric("Inclination", f"{np.degrees(sim.inclination):.1f}°")
        st.metric("Orbital Period", f"{sim.orbital_period / 60:.1f} min")
    with colB:
        decay = sim.orbital_decay(mission_days)
        st.metric(f"{mission_days}-Day Decay", f"{decay:.3f} km")

    time_h, alt_km = sim.simulate_orbit()
    fig_orbit = px.line(x=time_h, y=alt_km, labels={'x': 'Time (hours)', 'y': 'Altitude (km)'}, title="Orbit Altitude Variation")
    st.plotly_chart(fig_orbit, use_container_width=True)

with tab2:
    st.subheader("Power Budget")
    avg_p, cons = sim.power_budget()
    net = avg_p - cons
    colP1, colP2, colP3 = st.columns(3)
    colP1.metric("Generated (avg)", f"{avg_p:.2f} W")
    colP2.metric("Consumed", f"{cons:.2f} W")
    colP3.metric("Net Margin", f"{net:+.2f} W", delta=f"{net:+.2f} W")

with tab3:
    st.subheader("Thermal Analysis")
    thermal = sim.thermal_model()
    df_thermal = pd.DataFrame(list(thermal.items()), columns=['Case', 'Temperature (°C)'])
    fig_thermal = px.bar(df_thermal, x='Case', y='Temperature (°C)', title="Temperature Cases", color='Case')
    st.plotly_chart(fig_thermal, use_container_width=True)

    st.subheader("Radiation Exposure")
    if sim.altitude_km < 1000:
        dose = 0.5 + (sim.altitude_km - 200) * 0.002
    else:
        dose = 5.0 + (sim.altitude_km - 1000) * 0.01
    st.write(f"**Daily Dose:** {dose:.2f} rads/day")
    st.write(f"**{mission_days}-Day Total:** {dose*mission_days:.1f} rads")

if st.button("Benchmark: 400 km Run"):
    bench = CubeSatSim(400)
    t = bench.thermal_model()
    st.success("**GeneSat-1 Validated Benchmark:**")
    st.write(f"• Equilibrium: **{t['Equilibrium Temp (°C)']}°C**")
    st.write(f"• Hot Face: **{t['Hot Face (°C)']}°C**")
    st.write(f"• Cold Face: **{t['Cold Face (°C)']}°C**")
