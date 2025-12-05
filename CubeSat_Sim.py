import streamlit as st
import urllib.parse
import requests
from jose import jwt

import json
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt

# -------------------------
# Email / plan helpers
# -------------------------
def is_edu_email(email: str) -> bool:
    """Return True if email ends with .edu (case-insensitive)."""
    if not email:
        return False
    return email.strip().lower().endswith(".edu")



# =========================
# Stripe Payment Links (Phase 1: simple)
# =========================
# TODO: Paste in your real Stripe payment link URLs from the dashboard.

STD_MONTHLY_LINK = os.getenv("CATSIM_STD_MONTHLY_LINK", "https://buy.stripe.com/9B6fZha0B43ugPmbKF1RC00")
STD_YEARLY_LINK  = os.getenv("CATSIM_STD_YEARLY_LINK", "https://buy.stripe.com/4gM7sL7StcA09mU1611RC01")
PRO_MONTHLY_LINK = os.getenv("CATSIM_PRO_MONTHLY_LINK", "https://buy.stripe.com/9B68wPegR8jKeHe8yt1RC02")
PRO_YEARLY_LINK  = os.getenv("CATSIM_PRO_YEARLY_LINK", "https://buy.stripe.com/dRmbJ17St8jK9mUeWR1RC03")
ACADEMIC_LINK    = os.getenv("CATSIM_ACAD_LINK", "https://buy.stripe.com/dRm9AT7StarS56E7up1RC04")   # optional
DEPT_LINK        = os.getenv("CATSIM_DEPT_LINK", "https://buy.stripe.com/4gM5kD5Kl9nO7eMaGB1RC05")   # optional
BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "https://catsim-backend-prod.onrender.com"  # <-- your Render backend URL
)

def api_url(path: str) -> str:
    return f"{BACKEND_BASE_URL.rstrip('/')}{path}"



# =========================
# Environment: dev vs prod
# =========================
ENV = os.getenv("CATSIM_ENV", "prod")  # "dev" or "prod"
IS_DEV = ENV != "prod"

CONFIG = {
    "dev": {
        # Dev Auth0 app
        "AUTH0_DOMAIN": "dev-qwn3runpmc616as6.us.auth0.com",
        "AUTH0_CLIENT_ID": "c2tqv60NLCMmiN6fGeIWk5DpYnb4t5Vo",
        "AUTH0_CLIENT_SECRET": os.getenv(
            "AUTH0_CLIENT_SECRET_DEV",
            "E9BAjy7QLsJ0GSAYPMoBvb-vg7lMeLObKBqdsBupoQoVcVUHM75hmXOSDm2jzuw7",
        ),
        "AUTH0_CALLBACK_URL": "https://cubesimprov2-noruuoxdtsrjzdskhuobbr.streamlit.app",
        "SHOW_DEV_PLAN_SIM": True,
    },
    "prod": {
        # Prod Auth0 app (values you already have)
        "AUTH0_DOMAIN": "dev-qwn3runpmc616as6.us.auth0.com",
        "AUTH0_CLIENT_ID": "XvRZKwcHlcToRYGMMwnZiNjLnNzJmUmU",
        "AUTH0_CLIENT_SECRET": os.getenv(
            "AUTH0_CLIENT_SECRET_PROD",
            "y7Sn91jH3sR1seU5uWPJAM89BSmS-pXfPQPcfDLzt_K3Cu2fk-D0vzYnA2sE2lah",
        ),
        "AUTH0_CALLBACK_URL": "https://cubesimprov2-lt6hcgkvpdvygnwbktyqdg.streamlit.app",
        "SHOW_DEV_PLAN_SIM": False,
    },
}[ENV]


AUTH0_DOMAIN = CONFIG["AUTH0_DOMAIN"]
AUTH0_CLIENT_ID = CONFIG["AUTH0_CLIENT_ID"]
AUTH0_CLIENT_SECRET = CONFIG["AUTH0_CLIENT_SECRET"]
AUTH0_CALLBACK_URL = CONFIG["AUTH0_CALLBACK_URL"]
AUTH0_AUDIENCE = "https://catsim-backend-api"
APP_BASE_URL = "https://cubesimprov2-lt6hcgkvpdvygnwbktyqdg.streamlit.app/"


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="CATSIM — CubeSat Mission Simulator",
    page_icon="CATS_Logo.png",
    layout="wide"
)

# =========================
# Auth0 helpers (simple flow)
# =========================

def auth0_login_url():
    """Build the Auth0 login URL."""
    params = {
        "response_type": "code",
        "client_id": AUTH0_CLIENT_ID,
        "redirect_uri": AUTH0_CALLBACK_URL,
        "scope": "openid profile email",
        "audience": AUTH0_AUDIENCE,  # 🔑 ask Auth0 for a token for your API
    }
    return f"https://{AUTH0_DOMAIN}/authorize?" + urllib.parse.urlencode(params)


BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "https://catsim-backend.onrender.com"  # your real backend URL
)

def api_url(path: str) -> str:
    return f"{BACKEND_BASE_URL.rstrip('/')}{path}"

def get_billing_portal_url(user) -> str | None:
    """
    Ask backend for a Stripe Billing Portal URL for this user.
    """
    if not user:
        return None

    payload = {
        "user_id": user.get("sub"),      # Auth0 subject
        "email": user.get("email"),      # Auth0 email (helps find Stripe customer)
    }

    try:
        resp = requests.post(
            api_url("/create-portal-session"),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("url")
    except Exception as e:
        st.sidebar.error("Could not open billing portal. Please contact support if this persists.")
        if os.getenv("CATSIM_ENV", "dev") == "dev":
            # Show backend status + body for debugging
            try:
                st.sidebar.write("Debug portal:", resp.status_code, resp.text)
            except Exception:
                st.sidebar.write("Debug portal exception:", str(e))
        return None

def sync_user_with_backend(user):
    """
    Best-effort sync of the Auth0 user into the FastAPI backend.
    Creates/updates a row in the users table and wires up Stripe customer IDs.
    Never crashes the UI if the backend is down.
    """
    if not user:
        return

    payload = {
        "user_id": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
    }

    try:
        resp = requests.post(
            api_url("/sync-user"),
            json=payload,
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as e:
        # In production we stay quiet; in dev we surface a warning
        if os.getenv("CATSIM_ENV", "dev") == "dev":
            try:
                st.sidebar.warning(f"User sync failed: {e}")
            except Exception:
                pass


def auth0_logout_url():
    """Optional: Auth0 logout URL."""
    params = {
        "client_id": AUTH0_CLIENT_ID,
        "returnTo": AUTH0_CALLBACK_URL,
    }
    return f"https://{AUTH0_DOMAIN}/v2/logout?" + urllib.parse.urlencode(params)


def login_button():
    auth_url = auth0_login_url()
    # Use Streamlit's built-in link button for reliable redirect
    st.link_button("Sign in", auth_url)
    st.stop()


def _exchange_code_for_tokens(code: str):
    """Exchange authorization code for tokens via Auth0 /oauth/token."""
    token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "code": code,
        "redirect_uri": AUTH0_CALLBACK_URL,
        "audience": AUTH0_AUDIENCE,  # keep this
    }

    resp = requests.post(token_url, data=data)

    if resp.status_code != 200:
        # Show the full error from Auth0 so we can diagnose it
        raise RuntimeError(f"{resp.status_code} {resp.text}")

    return resp.json()



def get_user():
    """
    Returns the current user dict, or None.

    1) If user already in session_state -> return it.
    2) Else, if we have ?code=... in URL -> exchange it, store user, clear params.
    3) Else return None.
    """
    # Already logged in this session
    if "user" in st.session_state:
        return st.session_state["user"]

    # Get ?code=... from query params (string in modern Streamlit)
    params = st.query_params
    code = params.get("code")

    # st.query_params used to return lists; be robust to both
    if isinstance(code, list):
        code = code[0]

    if code:
        try:
            tokens = _exchange_code_for_tokens(code)
            id_token = tokens["id_token"]
            claims = jwt.get_unverified_claims(id_token)

            user = {
                "sub": claims.get("sub"),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "picture": claims.get("picture"),
            }
            st.session_state["user"] = user

            # Clear query params so the code is not reused on rerun
            st.query_params.clear()

            return user
        except Exception as e:
            st.error(f"Auth error: {e}")
            return None

    return None

def describe_subscription(sub_state: dict):
    """
    Turn backend subscription state into (label, end_date_text) for the UI.
    sub_state is what /subscription-state returns:
      {
        "plan_key": "pro_monthly",
        "human_readable": "Pro Monthly",
        "status": "active",
        "current_period_end": "2026-01-03T12:34:56+00:00" or None
      }
    """
    plan_key = sub_state.get("plan_key")
    status = (sub_state.get("status") or "").lower()
    human = sub_state.get("human_readable") or ""

    # No Stripe subscription → you're in dev trial / free
    if not plan_key:
        label = "Trial (Standard)"
        end_text = "Ends: 2026-01-03"  # keep your dev placeholder if you want
        return label, end_text

    # Map plan_key to UI labels
    if plan_key.startswith("pro_"):
        label = "Pro"
    elif plan_key.startswith("standard_"):
        label = "Standard"
    elif plan_key == "academic_yearly":
        label = "Academic Pro"
    elif plan_key == "dept_yearly":
        label = "Department License"
    else:
        label = human or plan_key

    # Show status (trialing vs active)
    if status == "trialing":
        label = f"Trial ({label})"

    # Format end date if present
    end_dt = sub_state.get("current_period_end")
    end_text = ""
    if end_dt:
        # end_dt may already be a datetime; if it's a string, parse date part
        if isinstance(end_dt, str):
            try:
                end_date = dt.date.fromisoformat(end_dt[:10])
            except Exception:
                end_date = None
        elif isinstance(end_dt, dt.datetime):
            end_date = end_dt.date()
        else:
            end_date = None

        if end_date:
            if status == "trialing":
                end_text = f"Trial ends: {end_date.isoformat()}"
            else:
                end_text = f"Renews: {end_date.isoformat()}"

    return label, end_text


def get_user():
    """
    Returns the current user dict, or None.

    1) If user already in session_state -> return it.
    2) Else, if we have ?code=... in URL -> exchange it, store user, clear params.
    3) Else return None.
    """
    # Already logged in this session
    if "user" in st.session_state:
        return st.session_state["user"]

    # Get ?code=... from query params (string in modern Streamlit)
    params = st.query_params
    code = params.get("code")

    # st.query_params used to return lists; be robust to both
    if isinstance(code, list):
        code = code[0]

    if code:
        try:
            tokens = _exchange_code_for_tokens(code)
            id_token = tokens["id_token"]
            claims = jwt.get_unverified_claims(id_token)

            user = {
                "sub": claims.get("sub"),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "picture": claims.get("picture"),
            }
            st.session_state["user"] = user

            # Clear query params so the code is not reused on rerun
            st.query_params.clear()

            return user
        except Exception as e:
            st.error(f"Auth error: {e}")
            return None

    return None

def start_checkout(tier: str):
    user = st.session_state.get("user")
    if not user:
        st.error("You must be signed in to upgrade.")
        return

    payload = {
        "user_id": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
        "tier": tier,  # "pro" or "standard"
    }

    try:
        resp = requests.post(
            api_url("/create-checkout-session"),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.write("Redirecting to Stripe Checkout...")
        st.markdown(
            f"<meta http-equiv='refresh' content='0; url={data['url']}'>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error("Checkout error: Could not start Stripe checkout.")
        if os.getenv("CATSIM_ENV", "dev") == "dev":
            try:
                st.write("Debug checkout:", resp.status_code, resp.text)
            except Exception:
                st.write("Debug checkout exception:", str(e))

def logout_button():
    # Auth0 logout URL – kills the Auth0 SSO session
    logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout?"
        f"client_id={AUTH0_CLIENT_ID}&"
        f"returnTo={urllib.parse.quote_plus(APP_BASE_URL)}"
    )

    if st.sidebar.button("Log out"):  # or wherever your logout button lives
        # 1) Clear Streamlit session
        st.session_state.clear()

        # 2) Clear URL query params (old code/state)
        st.experimental_set_query_params()

        # 3) Redirect the browser to Auth0 logout
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url={logout_url}">
            """,
            unsafe_allow_html=True,
        )
        st.stop()

def inject_brand_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #1d4ed8 !important;
            --primaryColor: #1d4ed8 !important;
        }

        /* Top nav bar */
        .cats-nav {
            background: #020617;
            padding: 0.4rem 1.2rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 1rem;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .cats-nav a {
            color: #e5e7eb;
            text-decoration: none;
            font-size: 0.9rem;
        }
        .cats-nav a:hover {
            color: #38bdf8;
        }

        /* Dataframe table sizing */
        .dataframe th, .dataframe td {
            font-size: 0.85rem !important;
        }

        /* PRIMARY buttons (e.g., Go Pro) — blue */
        button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background-color: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
            color: #ffffff !important;
        }
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #1e40af !important;
            border-color: #1e40af !important;
            color: #ffffff !important;
        }

        /* SIDEBAR non-primary buttons — white with blue border */
        div[data-testid="stSidebar"] button:not([data-testid="baseButton-primary"]) {
            background-color: #ffffff !important;
            color: #1d4ed8 !important;
            border: 1px solid #1d4ed8 !important;
        }
        div[data-testid="stSidebar"] button:not([data-testid="baseButton-primary"]):hover {
            background-color: #eff6ff !important;
            color: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def start_checkout(tier: str):
    user = st.session_state.get("user")
    if not user:
        st.error("You must be signed in to upgrade.")
        return

    payload = {
        "user_id": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
        "tier": tier,  # "pro" or "standard"
    }

    try:
        resp = requests.post(
            api_url("/create-checkout-session"),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.write("Redirecting to Stripe Checkout...")
        st.markdown(
            f"<meta http-equiv='refresh' content='0; url={data['url']}'>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error("Checkout error: Could not start Stripe checkout.")
        if os.getenv("CATSIM_ENV", "dev") == "dev":
            try:
                st.write("Debug checkout:", resp.status_code, resp.text)
            except Exception:
                st.write("Debug checkout exception:", str(e))




def show_header(user):
    col1, col2 = st.columns([1, 3])
    with col1:
        logo_paths = ["CATS_Logo.png", "assets/CATS_Logo.png"]
        shown = False
        for p in logo_paths:
            if os.path.exists(p):
                st.image(p, width=150)
                shown = True
                break
        if not shown:
            st.markdown("🛰️")
    with col2:
        name = user.get("name") if user else "Guest"
        env_label = "Development" if IS_DEV else "Production"
        st.markdown(
            f"""
            # **CATSIM — CubeSat Mission Simulator**
            #### by Cleveland Aerospace Technology Services  
            *Davidsonville, Maryland, USA*  

            <span style="font-size:0.85rem;color:#6b7280;">
            Signed in as: <strong>{name}</strong> • Environment: <strong>{env_label}</strong>
            </span>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="cats-nav">
            <a href="#catsim--cube-sat-mission-simulator">Home</a>
            <a href="https://www.clevelandaerospace.com" target="_blank" rel="noopener noreferrer">Company</a>
            <a href="mailto:press@clevelandaerospace.com">Press</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")


user = get_user()
if not user:
    st.title("CATSIM — Sign in")
    st.write("Please sign in to use the CubeSat Mission Simulator.")
    login_button()
    st.stop()

# Logged-in path from here down
inject_brand_css()
show_header(user)

# Sync Auth0 identity into backend DB (best-effort; non-fatal if it fails)
sync_user_with_backend(user)




# =========================
# Physical constants (SI)
# =========================
MU = 3.986004418e14
R_E = 6371e3
OMEGA_E = 7.2921159e-5
SIGMA = 5.670374419e-8
SOLAR_CONST = 1366.0
EARTH_IR = 237.0
ALBEDO = 0.3
DAY_SEC = 86400.0

# For debris lifetime modeling
R_EARTH_KM = R_E / 1000.0


def clamp_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def earth_view_factor(alt_m):
    r = R_E + float(alt_m)
    theta = np.arcsin(np.clip(R_E / r, -1.0, 1.0))
    return (1.0 - np.cos(theta)) / 2.0


def rho_msis_simple(h_m):
    H = 60e3
    rho_200 = 2.5e-11
    h = np.maximum(h_m, 200e3)
    return np.clip(rho_200 * np.exp(-(h - 200e3) / H), 1e-13, None)


def eci_to_ecef_xyz(x_eci, y_eci, z_eci, t):
    cosw = np.cos(OMEGA_E * t)
    sinw = np.sin(OMEGA_E * t)
    x_ecef = cosw * x_eci + sinw * y_eci
    y_ecef = -sinw * x_eci + cosw * y_eci
    z_ecef = z_eci
    return x_ecef, y_ecef, z_ecef


def sun_vector_eci(incl_rad, beta_deg):
    hhat = np.array([0.0, -np.sin(incl_rad), np.cos(incl_rad)])
    cb = np.cos(np.radians(beta_deg))
    sb = np.sin(np.radians(beta_deg))
    S = cb * np.array([1.0, 0.0, 0.0]) + sb * hhat
    S /= np.linalg.norm(S)
    return S


def eclipse_mask_from_vec(x_km, y_km, z_km, S_vec):
    r = np.vstack((x_km, y_km, z_km)).T
    S = np.asarray(S_vec, dtype=float)
    r_dot_S = r @ S
    r_sq = np.sum(r * r, axis=1)
    dist_ax_sq = r_sq - r_dot_S ** 2
    return (r_dot_S < 0.0) & (dist_ax_sq < (R_E / 1000.0) ** 2)


def eclipse_fraction_beta(alt_m, beta_deg):
    r = R_E + float(alt_m)
    Re_r = np.clip(R_E / r, 0.0, 1.0)
    root = np.sqrt(1.0 - Re_r ** 2)
    cb = np.cos(np.radians(beta_deg))
    if cb <= 0.0:
        return 0.0
    val = root / cb
    if val >= 1.0:
        return 0.0
    return float(np.arccos(val) / np.pi)


# =========================
# Heritage references
# =========================

GENESAT_REF = {
    "alt_km": 416.5,
    "inc_deg": 40.0,
    "period_min": 92.9,
    "mass_kg": 4.6,
    "oap_W": 4.5,
}

PHARMASAT_REF = {
    "alt_km": 410.0,
    "inc_deg": 40.0,
    "period_min": 92.8,
    "mass_kg": 4.5,
    "oap_W": 4.0,
}

OOREOS_REF = {
    "alt_km": 650.0,
    "inc_deg": 72.0,
    "period_min": 97.0,
    "mass_kg": 5.5,
    "oap_W": 5.0,
}

GENESAT_DEFAULTS = dict(
    altitude_km=GENESAT_REF["alt_km"],  # 416.5 km
    incl_deg=GENESAT_REF["inc_deg"],
    mass_kg=GENESAT_REF["mass_kg"],
    Cd=2.2,
    panel_area_m2=0.03,
    panel_eff=0.25,
    absorptivity=0.65,
    emissivity=0.83,
    target_avg_power_W=GENESAT_REF["oap_W"],
)


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
        self.T_orbit = 2 * np.pi * np.sqrt(self.r ** 3 / MU)
        self.n = 2 * np.pi / self.T_orbit
        self.v = np.sqrt(MU / self.r)
        self.VF = earth_view_factor(self.h)

        self.S = sun_vector_eci(self.i, self.beta_deg)

    def set_beta(self, beta_deg):
        self.beta_deg = float(beta_deg)
        self.S = sun_vector_eci(self.i, self.beta_deg)

    def eclipse_fraction(self):
        return eclipse_fraction_beta(self.h, self.beta_deg)

    def orbit_eci(self, N=720):
        t = np.linspace(0.0, self.T_orbit, N, endpoint=False)
        u = self.n * t
        r_km = self.r / 1000.0
        x = r_km * np.cos(u)
        y = r_km * np.sin(u) * np.cos(self.i)
        z = r_km * np.sin(u) * np.sin(self.i)
        return t, u, x, y, z

    def long_orbit_eci(self, num_orbits=10, N_per_orbit=720):
        total_pts = int(num_orbits * N_per_orbit)
        t = np.linspace(0.0, num_orbits * self.T_orbit, total_pts, endpoint=False)
        u = self.n * t
        r_km = self.r / 1000.0
        x = r_km * np.cos(u)
        y = r_km * np.sin(u) * np.cos(self.i)
        z = r_km * np.sin(u) * np.sin(self.i)
        return t, u, x, y, z

    def ground_track_from_eci(self, t, x_eci_km, y_eci_km, z_eci_km):
        x_m = x_eci_km * 1000.0
        y_m = y_eci_km * 1000.0
        z_m = z_eci_km * 1000.0
        x_ecef, y_ecef, z_ecef = eci_to_ecef_xyz(x_m, y_m, z_m, t)
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))
        lat = np.degrees(np.arcsin(z_ecef / np.sqrt(x_ecef ** 2 + y_ecef ** 2 + z_ecef ** 2)))
        lon = clamp_angle_deg(lon)
        return lon, lat

    def instantaneous_power(self, attitude, t, x_km, y_km, z_km, A_panel, eta):
        S = self.S
        r = np.vstack((x_km, y_km, z_km)).T
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
        ecl = eclipse_mask_from_vec(x_km, y_km, z_km, S)
        cos_inc = cos_inc * (~ecl)
        P = SOLAR_CONST * A_panel * eta * cos_inc
        return P, cos_inc, ecl

    def avg_power(self, attitude, A_panel, eta):
        t, u, x, y, z = self.orbit_eci(N=1440)
        P, _, _ = self.instantaneous_power(attitude, t, x, y, z, A_panel, eta)
        return float(P.mean())

    def avg_power_at_beta(self, attitude, beta_deg, A_panel, eta, N=720):
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

    def thermal_equilibrium(self, A_abs=None, A_rad=None, Q_internal_W=0.0, beta_for_thermal=None):
        if A_abs is None:
            A_abs = self.A_panel
        if A_rad is None:
            A_rad = 6.0 * self.A_panel

        beta_local = self.beta_deg if beta_for_thermal is None else float(beta_for_thermal)
        efrac = eclipse_fraction_beta(self.h, beta_local)
        VF = self.VF

        Q_solar = SOLAR_CONST * self.alpha * A_abs * (1.0 - efrac)
        Q_albedo = ALBEDO * SOLAR_CONST * self.alpha * A_abs * VF * (1.0 - efrac)
        Q_ir = EARTH_IR * self.eps * A_abs * VF
        Q_total = Q_solar + Q_albedo + Q_ir + float(Q_internal_W)

        T_K = (Q_total / (self.eps * SIGMA * A_rad)) ** 0.25
        return float(T_K - 273.15), Q_solar, Q_albedo, Q_ir, float(Q_internal_W), float(Q_total)

    def thermal_equilibrium_2node(
        self,
        A_abs_shell=None,
        A_rad_shell=None,
        Q_int_shell_W=0.0,
        Q_int_interior_W=5.0,
        k_cond_W_per_K=0.5,
        beta_for_thermal=None,
        T_shell_init_C=0.0,
        T_int_init_C=0.0,
        max_iter=200,
        tol_K=1e-3,
    ):
        """
        Simple 2-node, orbit-average thermal model:
        - Shell node sees solar, albedo, IR & radiates to space.
        - Interior node receives internal dissipation and conducts to shell.
        Returns (T_shell_C, T_int_C, diagnostics).
        """

        import math  # local import OK

        # --- Areas ---
        if A_abs_shell is None:
            A_abs_shell = self.A_panel
        if A_rad_shell is None:
            A_rad_shell = 6.0 * self.A_panel

        # --- β and eclipse fraction ---
        beta_local = self.beta_deg if beta_for_thermal is None else float(beta_for_thermal)
        efrac = eclipse_fraction_beta(self.h, beta_local)
        VF = self.VF

        # --- Orbit-average fluxes on shell ---
        Q_solar_shell = SOLAR_CONST * self.alpha * A_abs_shell * (1.0 - efrac)
        Q_albedo_shell = ALBEDO * SOLAR_CONST * self.alpha * A_abs_shell * VF * (1.0 - efrac)
        Q_ir_shell = EARTH_IR * self.eps * A_abs_shell * VF

        Q_env_shell = Q_solar_shell + Q_albedo_shell + Q_ir_shell

        Q_int_shell = float(Q_int_shell_W)
        Q_int_int = float(Q_int_interior_W)

        # --- Initial guesses in Kelvin ---
        T_shell = T_shell_init_C + 273.15
        T_int = T_int_init_C + 273.15

        converged = False

        # --- Iterate ---
        for it in range(max_iter):

            # Radiation from shell to space
            Q_rad_shell = self.eps * SIGMA * A_rad_shell * (T_shell**4)

            # Conduction between nodes
            Q_cond_shell = k_cond_W_per_K * (T_int - T_shell)
            Q_cond_int = -Q_cond_shell

            # Residual equations
            R_shell = Q_env_shell + Q_int_shell + Q_cond_shell - Q_rad_shell
            R_int = Q_int_int + Q_cond_int

            # Jacobians (dQ/dT)
            dQdT_shell = (
                4.0 * self.eps * SIGMA * A_rad_shell * (T_shell**3)
                + k_cond_W_per_K
                + 1e-6
            )
            dQdT_int = k_cond_W_per_K + 1e-6

            # Newton steps
            dT_shell = R_shell / dQdT_shell
            dT_int = R_int / dQdT_int

            T_shell_new = T_shell + dT_shell
            T_int_new = T_int + dT_int

            # Convergence check
            if max(abs(T_shell_new - T_shell), abs(T_int_new - T_int)) < tol_K:
                T_shell = T_shell_new
                T_int = T_int_new
                converged = True
                break

            T_shell = T_shell_new
            T_int = T_int_new

        # Final heat terms
        Q_rad_shell = self.eps * SIGMA * A_rad_shell * (T_shell**4)
        Q_cond_shell = k_cond_W_per_K * (T_int - T_shell)
        Q_cond_int = -Q_cond_shell

        diagnostics = {
            "Q_solar_shell_W": Q_solar_shell,
            "Q_albedo_shell_W": Q_albedo_shell,
            "Q_ir_shell_W": Q_ir_shell,
            "Q_env_shell_W": Q_env_shell,
            "Q_int_shell_W": Q_int_shell,
            "Q_int_interior_W": Q_int_int,
            "Q_rad_shell_W": Q_rad_shell,
            "Q_cond_shell_W": Q_cond_shell,
            "Q_cond_interior_W": Q_cond_int,
            "iterations": it + 1,
            "converged": converged,
            "beta_deg": beta_local,
            "eclipse_fraction": efrac,
            "VF": VF,
        }

        return (T_shell - 273.15), (T_int - 273.15), diagnostics

    def drag_decay_days(self, days, A_drag=None):
        A = self.A_panel if A_drag is None else float(A_drag)
        a = R_E + self.h
        out = []
        for _ in range(int(days)):
            rho = rho_msis_simple(a - R_E)
            da_dt = - (self.Cd * A / self.m) * np.sqrt(MU * a) * rho
            a = a + da_dt * DAY_SEC
            if a < R_E + 120e3:
                a = R_E + 120e3
            out.append(a - R_E)
        return np.array(out) / 1000.0


# =========================
# NASA debris lifetime model + ODAR helpers
# =========================

LIFETIME_REF_TABLE = [
    (350, 0.10),
    (375, 0.20),
    (400, 0.50),
    (425, 1.0),
    (450, 1.8),
    (475, 3.0),
    (500, 4.5),
    (525, 7.0),
    (550, 10.0),
    (575, 17.0),
    (600, 25.0),
    (625, 40.0),
    (650, 60.0),
    (675, 100.0),
    (700, 150.0),
    (725, 250.0),
    (750, 400.0),
    (775, 650.0),
    (800, 900.0),
]

BC_REF_KG_M2 = 50.0  # reference ballistic coefficient


def _interp_lifetime_base(perigee_alt_km: float) -> float:
    import math

    if perigee_alt_km <= LIFETIME_REF_TABLE[0][0]:
        h0, t0 = LIFETIME_REF_TABLE[0]
        h1, t1 = LIFETIME_REF_TABLE[1]
        logt0 = math.log(t0)
        logt1 = math.log(t1)
        slope = (logt1 - logt0) / (h1 - h0)
        logt = logt0 + slope * (perigee_alt_km - h0)
        return max(math.exp(logt), 0.01)

    if perigee_alt_km >= LIFETIME_REF_TABLE[-1][0]:
        h0, t0 = LIFETIME_REF_TABLE[-2]
        h1, t1 = LIFETIME_REF_TABLE[-1]
        logt0 = math.log(t0)
        logt1 = math.log(t1)
        slope = (logt1 - logt0) / (h1 - h0)
        logt = logt1 + slope * (perigee_alt_km - h1)
        return math.exp(logt)

    for (h0, t0), (h1, t1) in zip(LIFETIME_REF_TABLE[:-1], LIFETIME_REF_TABLE[1:]):
        if h0 <= perigee_alt_km <= h1:
            logt0 = math.log(t0)
            logt1 = math.log(t1)
            f = (perigee_alt_km - h0) / (h1 - h0)
            logt = logt0 + f * (logt1 - logt0)
            return math.exp(logt)

    return 25.0


def estimate_orbital_lifetime_years(
    perigee_alt_km: float,
    mass_kg: float | None = None,
    cross_section_m2: float | None = None,
    cd: float = 2.2,
    solar_activity_scale: float = 1.0,
) -> float:
    import math

    if perigee_alt_km < 150.0:
        return 0.0

    base_life = _interp_lifetime_base(perigee_alt_km)

    if mass_kg is not None and cross_section_m2 is not None and cross_section_m2 > 0.0:
        bc = mass_kg / (cd * cross_section_m2)
        bc_scale = bc / BC_REF_KG_M2
    else:
        bc_scale = 1.0

    if solar_activity_scale <= 0.0:
        solar_activity_scale = 1.0

    lifetime = base_life * bc_scale / solar_activity_scale
    return lifetime


def nasa_debris_compliance_check(
    sma_km: float,
    ecc: float,
    mass_kg: float | None = None,
    cross_section_m2: float | None = None,
    cd: float = 2.2,
    solar_activity_scale: float = 1.0,
) -> dict:
    r_p_km = sma_km * (1.0 - ecc)
    h_p_km = r_p_km - R_EARTH_KM

    lifetime_years = estimate_orbital_lifetime_years(
        perigee_alt_km=h_p_km,
        mass_kg=mass_kg,
        cross_section_m2=cross_section_m2,
        cd=cd,
        solar_activity_scale=solar_activity_scale,
    )

    if lifetime_years <= 25.0:
        status = "Compliant"
        emoji = "✅"
        note = (
            "Estimated post-mission orbital lifetime is ≤ 25 years, consistent with "
            "NASA's LEO post-mission disposal guideline (NPR 8715.6 / NASA-STD-8719.14)."
        )
    elif 25.0 < lifetime_years <= 35.0:
        status = "Borderline"
        emoji = "⚠️"
        note = (
            "Estimated lifetime is slightly above 25 years. Compliance will depend on "
            "detailed atmospheric modeling, ballistic coefficient, and solar cycle "
            "assumptions; an active disposal strategy may still be required."
        )
    else:
        status = "Not Compliant"
        emoji = "❌"
        note = (
            "Estimated post-mission orbital lifetime significantly exceeds 25 years. "
            "NASA debris mitigation policy would typically require an active disposal "
            "strategy (e.g., deorbit burn, drag augmentation device, or lower disposal orbit)."
        )

    return {
        "perigee_alt_km": h_p_km,
        "lifetime_years": lifetime_years,
        "status": status,
        "emoji": emoji,
        "note": note,
    }


def generate_odar_summary_text(
    mission_name: str,
    sma_km: float,
    ecc: float,
    inc_deg: float | None,
    mass_kg: float | None,
    cross_section_m2: float | None,
    mission_life_years: float | None,
    disposal_mode: str,
    compliance_result: dict,
) -> str:
    h_p = compliance_result["perigee_alt_km"]
    t_life = compliance_result["lifetime_years"]
    status = compliance_result["status"]

    inc_str = f" at {inc_deg:.1f}° inclination" if inc_deg is not None else ""
    mass_str = f" with a dry mass of ~{mass_kg:.1f} kg" if mass_kg is not None else ""
    area_str = (
        f" and an estimated average cross-sectional area of ~{cross_section_m2:.3f} m²"
        if cross_section_m2 is not None
        else ""
    )
    life_str = (
        f" The planned operational lifetime is approximately {mission_life_years:.1f} years."
        if mission_life_years is not None
        else ""
    )

    disposal_sentence = (
        f" At end-of-mission, the spacecraft will be left in (or maneuvered to) an orbit "
        f"with a perigee altitude of approximately {h_p:.0f} km, following the disposal mode: "
        f"{disposal_mode}."
    )

    if status == "Compliant":
        compliance_sentence = (
            f" Using the CATSIM orbital lifetime model, which is calibrated to typical "
            f"small spacecraft ballistic coefficients and standard atmospheric references, "
            f"the post-mission orbital lifetime is estimated to be about {t_life:.1f} years. "
            f"This is less than the 25-year guideline for LEO post-mission disposal established "
            f"in NASA NPR 8715.6 and NASA-STD-8719.14; therefore, the mission is expected to be "
            f"**compliant** with NASA's LEO orbital debris mitigation requirements for post-mission disposal."
        )
    elif status == "Borderline":
        compliance_sentence = (
            f" Using the CATSIM orbital lifetime model, the post-mission orbital lifetime is "
            f"estimated to be about {t_life:.1f} years, which is slightly above the 25-year "
            f"guideline in NASA NPR 8715.6 and NASA-STD-8719.14. Compliance will depend on the "
            f"actual on-orbit ballistic coefficient and solar activity cycle. A margin analysis "
            f"and/or an active disposal maneuver is recommended to ensure the lifetime remains "
            f"within 25 years under conservative assumptions."
        )
    else:
        compliance_sentence = (
            f" Using the CATSIM orbital lifetime model, the post-mission orbital lifetime is "
            f"estimated to be about {t_life:.1f} years, which significantly exceeds the 25-year "
            f"guideline in NASA NPR 8715.6 and NASA-STD-8719.14. As currently defined, this "
            f"mission would **not be compliant** with NASA's LEO post-mission disposal "
            f"requirements. An active disposal strategy (e.g., perigee-lowering burn or drag "
            f"augmentation device) will be required to reduce the orbital lifetime below 25 years."
        )

    intro = (
        f"{mission_name} is a small satellite mission in low Earth orbit with an end-of-mission "
        f"orbit characterized by a semi-major axis of approximately {sma_km:.0f} km and an "
        f"eccentricity of {ecc:.4f}{inc_str}.{mass_str}{area_str}{life_str}"
    )

    closing = (
        " These estimates are intended for design and trade studies and should be validated "
        "against an approved orbital debris assessment tool (e.g., NASA DAS) during formal "
        "mission reviews."
    )

    return intro + "\n\n" + disposal_sentence + "\n\n" + compliance_sentence + "\n\n" + closing


# =========================
# Pricing model & plan state (via backend)
# =========================

# Attach Auth0 identity early
auth_email = (user or {}).get("email", "unknown")
auth_name = (user or {}).get("name", "")
auth_pic = (user or {}).get("picture")
auth_sub = (user or {}).get("sub", "")

today = dt.date.today()

# Ensure we have a trial_start baseline for local trial fallback
if "trial_start" not in st.session_state:
    st.session_state.trial_start = today.isoformat()

# One-time fetch of raw subscription state from backend, keyed by Auth0 sub
if "subscription_state" not in st.session_state:
    st.session_state.subscription_state = fetch_subscription_state(auth_sub)

sub_raw = st.session_state.subscription_state
sub = normalize_subscription_state(sub_raw)  # <-- unwrap + normalize

trial_start = dt.date.fromisoformat(st.session_state.trial_start)
trial_end = trial_start + dt.timedelta(days=30)

backend_plan = None
status = "active"
customer_id = None

if sub:
    # Handle either 'plan' or 'plan_key'
    plan_key = (sub.get("plan_key") or "").lower()
    plan_field = (sub.get("plan") or "").lower()

    if plan_field in ("standard", "pro"):
        backend_plan = plan_field
    elif plan_key.startswith("pro_"):
        backend_plan = "pro"
    elif plan_key.startswith("standard_"):
        backend_plan = "standard"
    elif plan_key in ("pro", "standard"):
        backend_plan = plan_key

    status = (sub.get("status") or "active")
    customer_id = sub.get("customer_id")

# If backend has a real subscription → trust it
if backend_plan:
    plan_base = backend_plan          # "standard" or "pro"
    in_trial = (status.lower() == "trialing")
    plan_effective = plan_base        # Pro tabs & export check this
else:
    # No Stripe subscription in DB (or backend unreachable) → local 30-day trial
    if "plan_base" not in st.session_state:
        st.session_state.plan_base = "trial"

    plan_base = st.session_state.plan_base
    in_trial = (plan_base == "trial") and (today <= trial_end)
    plan_effective = "standard" if in_trial else plan_base
    status = "active"
    customer_id = None

# Persist into session_state for the rest of the app
st.session_state.plan_base = plan_base
st.session_state.effective_plan = plan_effective
st.session_state.in_trial = in_trial
st.session_state.trial_start = trial_start.isoformat()
st.session_state.trial_end = trial_end.isoformat()
st.session_state.stripe_customer_id = customer_id




# ----- SIDEBAR: account + plan + sim setup -----
with st.sidebar:
    # -------------------------
    # Account
    # -------------------------
    st.markdown("### Account")

    auth_email = (user or {}).get("email", "unknown")
    auth_name = (user or {}).get("name", "")
    auth_pic = (user or {}).get("picture")

    if auth_pic:
        st.image(auth_pic, width=64)
    if auth_name:
        st.markdown(f"**{auth_name}**")
    st.caption(auth_email)

    st.header("Plan & Billing")
    
    st.markdown("### Account")

    user = st.session_state.get("user")  # however you store Auth0 user

    if user:
        portal_url = get_billing_portal_url(user)
        if portal_url:
            st.link_button(
                "Manage billing & invoices",
                portal_url,
                use_container_width=True,
                type="secondary",
            )
    else:
        st.caption("Sign in to manage your subscription.")
    # Dev-only simulated plan controls
    if CONFIG.get("SHOW_DEV_PLAN_SIM", False):
        st.markdown(
            f"<meta http-equiv='refresh' content='0; url={logout_url}'>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Avatar + name
    acct_col1, acct_col2 = st.columns([1, 2])
    with acct_col1:
        if auth_pic:
            st.image(auth_pic, width=64)
        else:
            st.markdown("🛰️")
    with acct_col2:
        if auth_name:
            st.markdown(f"**{auth_name}**")
        st.caption(auth_email)

    st.divider()

 # -------------------------
    # Plan & Billing
    # -------------------------
    st.markdown("### Plan & Billing")

    # Use effective plan + normalized subscription to drive the label
    sub_norm = sub  # already normalized by normalize_subscription_state
    status = (sub_norm.get("status") or "").lower() if sub_norm else ""
    human = (sub_norm.get("human_readable") or "").strip() if sub_norm else ""
    end_dt_str = sub_norm.get("current_period_end") if sub_norm else None

    # Decide label based on effective plan + trial state
    if plan_effective == "pro":
        label = human or "Pro"
    elif plan_effective == "standard":
        if in_trial:
            label = "Trial (Standard)"
        else:
            label = human or "Standard"
    else:
        # No backend sub; local trial or fully free
        if in_trial:
            label = "Trial (Standard)"
        else:
            label = "Free (Standard features only)"

    # Decide end / renew text
    end_text = ""
    if end_dt_str:
        try:
            end_date = dt.date.fromisoformat(end_dt_str[:10])
            if status == "trialing" or (in_trial and plan_effective != "pro"):
                end_text = f"Trial ends: {end_date.isoformat()}"
            else:
                end_text = f"Renews: {end_date.isoformat()}"
        except Exception:
            end_text = ""
    elif in_trial and plan_effective != "pro":
        # Local 30-day trial fallback
        end_text = f"Trial ends: {trial_end.isoformat()}"

    

    st.markdown(
        f"""
        <div style="padding: 12px; background: #f5f9ff; border: 1px solid #c3d5ff;">
            <b>Current plan:</b> 🚀 {label}<br>
            {end_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

       # -------------------------
    # Upgrade buttons
    # -------------------------
    if plan_effective != "pro":
        st.warning("Pro features are locked. Upgrade to Pro to enable.")

        st.markdown("### Upgrade")

        col_a, col_b = st.columns(2)

        # Standard plans
        with col_a:
            if st.button("Standard $9.99/mo"):
                with st.spinner("Contacting secure Stripe checkout..."):
                    url = create_checkout_session("standard_monthly", auth_email)
                if url:
                    st.session_state.checkout_url = url

            if st.button("Standard $99/yr"):
                with st.spinner("Contacting secure Stripe checkout..."):
                    url = create_checkout_session("standard_yearly", auth_email)
                if url:
                    st.session_state.checkout_url = url

        # Pro plans
        with col_b:
            if st.button("🚀 Pro $19.99/mo", type="primary"):
                with st.spinner("Contacting secure Stripe checkout..."):
                    url = create_checkout_session("pro_monthly", auth_email)
                if url:
                    st.session_state.checkout_url = url

            if st.button("🚀 Pro $199/yr", type="primary"):
                with st.spinner("Contacting secure Stripe checkout..."):
                    url = create_checkout_session("pro_yearly", auth_email)
                if url:
                    st.session_state.checkout_url = url

        # -------------------------
        # Education
        # -------------------------
        st.markdown("**Education:**")
        col_c, col_d = st.columns(2)

        # Check if the signed-in user has a .edu email
        edu_ok = is_edu_email(auth_email)

        with col_c:
            if not edu_ok:
                # Disabled Academic button + explanation
                st.button(
                    "Academic Pro $99/yr (.edu only)",
                    key="academic_disabled",
                    help="Academic license (requires .edu email)",
                    disabled=True,
                )
                st.caption(
                    f"Academic pricing is reserved for users with a valid .edu email address. "
                    f"Current email: `{auth_email}`"
                )
            else:
                # Only .edu users can actually start Academic checkout
                if st.button(
                    "Academic Pro $99/yr",
                    key="academic",
                    help="Academic license",
                ):
                    with st.spinner("Contacting secure Stripe checkout..."):
                        url = create_checkout_session("academic_yearly", auth_email)
                    if url:
                        st.session_state.checkout_url = url

        
    else:
        st.success("✅ You are on the Pro plan.")


    # Checkout link (still in sidebar)
    if st.session_state.get("checkout_url"):
        st.markdown("### Checkout / Billing")
        st.info(
            "✅ Click below to open secure Stripe checkout / billing. "
            "After completing payment, refresh CATSIM to update your plan."
        )
        st.markdown(f"[Open Stripe]({st.session_state.checkout_url})")

    # -------------------------
    # Simulation setup
    # -------------------------
    st.divider()
    st.markdown("### Simulation setup")

    # Preset & validation
    with st.expander("Preset & validation", expanded=True):
        use_genesat = st.checkbox("Load GeneSat-1 defaults", True)
        auto_cal = st.checkbox("Calibrate avg power to GeneSat target (~4.5 W)", True)

    # Mission parameters
    st.markdown("#### Mission parameters")

    if use_genesat:
        altitude_km = st.slider(
            "Altitude (km)", 200.0, 700.0, GENESAT_DEFAULTS["altitude_km"]
        )
        incl_deg = st.slider(
            "Inclination (deg)", 0.0, 98.0, GENESAT_DEFAULTS["incl_deg"]
        )
        mass_kg = st.number_input(
            "Mass (kg)", 0.1, 50.0, GENESAT_DEFAULTS["mass_kg"], 0.1
        )
        Cd = st.slider(
            "Drag coefficient Cd", 1.5, 3.0, GENESAT_DEFAULTS["Cd"], 0.1
        )
        panel_area = st.number_input(
            "Panel area / face (m²)",
            0.001,
            0.5,
            GENESAT_DEFAULTS["panel_area_m2"],
            0.001,
        )
        panel_eff = st.slider(
            "Panel efficiency η", 0.05, 0.38, GENESAT_DEFAULTS["panel_eff"], 0.01
        )
        absorp = st.slider(
            "Absorptivity α", 0.1, 1.0, GENESAT_DEFAULTS["absorptivity"], 0.01
        )
        emiss = st.slider(
            "Emissivity ε", 0.1, 1.0, GENESAT_DEFAULTS["emissivity"], 0.01
        )
        target_avgW = GENESAT_DEFAULTS["target_avg_power_W"]
    else:
        altitude_km = st.slider("Altitude (km)", 200.0, 2000.0, 500.0)
        incl_deg = st.slider("Inclination (deg)", 0.0, 98.0, 51.6)
        mass_kg = st.number_input("Mass (kg)", 0.1, 200.0, 4.0, 0.1)
        Cd = st.slider("Drag coefficient Cd", 1.0, 3.5, 2.2, 0.1)
        panel_area = st.number_input(
            "Panel area / face (m²)", 0.001, 2.0, 0.05, 0.001
        )
        panel_eff = st.slider("Panel efficiency η", 0.05, 0.38, 0.28, 0.01)
        absorp = st.slider("Absorptivity α", 0.1, 1.0, 0.6, 0.01)
        emiss = st.slider("Emissivity ε", 0.1, 1.0, 0.8, 0.01)
        target_avgW = st.number_input(
            "Target avg power (W) for calibration", 0.1, 50.0, 4.5, 0.1
        )

    # Attitude & ops
    st.markdown("#### Attitude & ops")

    attitude = st.radio(
        "Attitude", ["body-spin", "sun-tracking", "nadir-pointing"], horizontal=False
    )
    elec_derate = st.slider(
        "Electrical derate (BOL→EOL, MPPT, wiring)", 0.40, 1.00, 0.70, 0.01
    )
    beta_deg = st.slider(
        "β-angle (deg) — Sun vs. orbital plane", -80.0, 80.0, 0.0, 0.5
    )
    show_play = st.checkbox("Show Play buttons on plots", True)
    anim_speed = st.slider("Animation speed (Plotly)", 0.1, 5.0, 1.0, 0.1)
    mission_days = st.slider("Mission duration (days)", 1, 365, 60)



# =========================
# Build simulator & calibration
# =========================
sim = CubeSatSim(
    altitude_km=altitude_km,
    incl_deg=incl_deg,
    mass_kg=mass_kg,
    Cd=Cd,
    panel_area_m2=panel_area,
    panel_eff=panel_eff,
    absorptivity=absorp,
    emissivity=emiss,
    beta_deg=beta_deg,
)
sim.set_beta(beta_deg)

# cal_factor is chosen so that the orbit-average power from the model
# is forced to match target_avgW BEFORE derate (when auto_cal is enabled).
cal_factor = 1.0
if auto_cal:
    P_now = sim.avg_power(attitude, sim.A_panel, sim.eta)
    if P_now > 1e-9:
        cal_factor = target_avgW / P_now

# =========================
# Tabs (Verification included)
# =========================
tab_orbit, tab_power, tab_thermal, tab_drag, tab_verify, tab_adv, tab_io = st.tabs(
    [
        "3D + Ground Track (aligned)",
        "Power",
        "Thermal",
        "Drag",
        "Verification (Heritage)",
        "Advanced Analysis (Pro)",
        "Save/Load & Export (Pro)",
    ]
)

# --- TAB 1: ORBIT ---
with tab_orbit:
    st.subheader("3D Orbit (ECI) + Aligned Ground Track (ECEF)")

    # Green pill-style heritage badge
    st.markdown(
        """
        <div style="display:inline-block; padding:0.25rem 0.75rem;
                    border-radius:999px; background-color:rgba(22,163,74,0.12);
                    color:#16a34a; font-size:0.8rem; margin-bottom:0.5rem;">
            ✔ Model anchored to GeneSat-1 / PharmaSat / O/OREOS flight data
        </div>
        """,
        unsafe_allow_html=True,
    )

    t, u, x_km, y_km, z_km = sim.orbit_eci(N=720)
    lon_deg, lat_deg = sim.ground_track_from_eci(t, x_km, y_km, z_km)
    eclipsed = eclipse_mask_from_vec(x_km, y_km, z_km, sim.S)

    x_sun, y_sun, z_sun = x_km.copy(), y_km.copy(), z_km.copy()
    x_sun[eclipsed] = None
    y_sun[eclipsed] = None
    z_sun[eclipsed] = None
    x_ecl, y_ecl, z_ecl = x_km.copy(), y_km.copy(), z_km.copy()
    x_ecl[~eclipsed] = None
    y_ecl[~eclipsed] = None
    z_ecl[~eclipsed] = None

    fig3d = go.Figure()
    uu = np.linspace(0, 2 * np.pi, 60)
    vv = np.linspace(0, np.pi, 30)
    UU, VV = np.meshgrid(uu, vv)
    xE = (R_E / 1000.0) * np.cos(UU) * np.sin(VV)
    yE = (R_E / 1000.0) * np.sin(UU) * np.sin(VV)
    zE = (R_E / 1000.0) * np.cos(VV)
    fig3d.add_trace(
        go.Surface(x=xE, y=yE, z=zE, opacity=0.35, showscale=False, name="Earth")
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=x_sun,
            y=y_sun,
            z=z_sun,
            mode="lines",
            line=dict(color="gold", width=4),
            name="Sunlit",
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=x_ecl,
            y=y_ecl,
            z=z_ecl,
            mode="lines",
            line=dict(color="gray", width=4),
            name="Eclipse",
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=[x_km[0]],
            y=[y_km[0]],
            z=[z_km[0]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Sat",
        )
    )

    frames3d = []
    step = 4
    for k in range(0, len(x_km), step):
        frames3d.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scatter3d(
                        x=[x_km[k]],
                        y=[y_km[k]],
                        z=[z_km[k]],
                        mode="markers",
                        marker=dict(size=6, color="red"),
                    )
                ],
                traces=[3],
            )
        )
    fig3d.frames = frames3d
    fig3d.update_layout(
        scene=dict(aspectmode="data"),
        height=560,
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep-earth",
    )
    if show_play:
        fig3d.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    x=0.05,
                    xanchor="left",
                    y=0.05,
                    yanchor="bottom",
                    pad={"r": 0, "t": 0},
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=int(40 / anim_speed), redraw=True
                                    ),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        )
    st.plotly_chart(fig3d, use_container_width=True)

    fig_gt = go.Figure()
    fig_gt.add_trace(
        go.Scattergeo(
            lon=lon_deg,
            lat=lat_deg,
            mode="lines",
            line=dict(color="royalblue", width=2),
            name="Path",
        )
    )
    fig_gt.add_trace(
        go.Scattergeo(
            lon=[lon_deg[0]],
            lat=[lat_deg[0]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Sat",
        )
    )
    frames_gt = []
    for k in range(0, len(lon_deg), step):
        frames_gt.append(
            go.Frame(
                name=str(k),
                data=[
                    go.Scattergeo(
                        lon=[lon_deg[k]],
                        lat=[lat_deg[k]],
                        mode="markers",
                        marker=dict(size=6, color="red"),
                    )
                ],
                traces=[1],
            )
        )
    fig_gt.frames = frames_gt
    fig_gt.update_layout(
        geo=dict(projection_type="natural earth", showland=True),
        height=360,
        margin=dict(t=10),
        uirevision="keep-gt",
    )
    if show_play:
        fig_gt.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    x=0.05,
                    xanchor="left",
                    y=0.05,
                    yanchor="bottom",
                    pad={"r": 0, "t": 0},
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=int(40 / anim_speed), redraw=True
                                    ),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        )
    st.plotly_chart(fig_gt, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Orbital period (min)", f"{sim.T_orbit / 60.0:.2f}")
    c2.metric("Eclipse fraction", f"{sim.eclipse_fraction():.3f}")
    c3.metric("Earth view factor", f"{sim.VF:.3f}")
    st.caption(
        "Tip: increasing |β| shortens/eradicates eclipse; beyond a critical β there's no eclipse."
    )


# --- TAB 2: POWER ---
with tab_power:
    st.subheader("Power — instantaneous, average, β-sweep, and SoC")
    st.caption("Orbit-average power is usually 30–60% of peak. GeneSat-1 ≈ 4–5 W OAP.")

    N_orbit = 720
    t_orb, u, x_km, y_km, z_km = sim.orbit_eci(N=N_orbit)
    dt_step = float(t_orb[1] - t_orb[0])
    P_inst, cos_inc, ecl = sim.instantaneous_power(
        attitude, t_orb, x_km, y_km, z_km, sim.A_panel, sim.eta
    )
    P_inst = P_inst * cal_factor * elec_derate

    st.plotly_chart(
        px.line(
            x=(t_orb / t_orb[-1]) * 360.0,
            y=P_inst,
            labels={"x": "Orbit phase (deg)", "y": "Power (W)"},
            title=(
                f"Instantaneous Power — {attitude} "
                f"(cal {cal_factor:.2f} × derate {elec_derate:.2f} @ β={beta_deg:.1f}°)"
            ),
        ),
        use_container_width=True,
    )

    st.markdown("### β-angle sweep (current attitude)")
    betas = np.linspace(-80, 80, 161)
    oap = np.array([sim.avg_power_at_beta(attitude, b, sim.A_panel, sim.eta) for b in betas])
    oap_scaled = oap * cal_factor * elec_derate
    df_beta = pd.DataFrame({"beta_deg": betas, "OAP_W": oap_scaled})
    st.plotly_chart(
        px.line(
            df_beta,
            x="beta_deg",
            y="OAP_W",
            title=f"Orbit-average Power vs β-angle ({attitude})",
            labels={"beta_deg": "β (deg)", "OAP_W": "Orbit-average Power (W)"},
        ),
        use_container_width=True,
    )

    st.markdown("### Battery & Load (baseline SoC)")
    cons_W = st.slider("Average consumption (W)", 0.1, 50.0, 3.0, 0.1)
    batt_Wh = st.slider("Battery capacity (Wh)", 5.0, 1000.0, 30.0, 1.0)
    start_soc = st.slider("Start SoC (%)", 0, 100, 50)
    eta_chg = st.slider("Charge efficiency η_c", 0.50, 1.00, 0.95, 0.01)
    eta_dis = st.slider("Discharge efficiency η_d", 0.50, 1.00, 0.95, 0.01)
    limit_charge = st.checkbox("Limit charge power", False)
    P_chg_max = (
        st.slider("Max charge power (W)", 1.0, 200.0, 30.0, 1.0) if limit_charge else None
    )

    total_steps = int(np.ceil((mission_days * DAY_SEC) / dt_step))
    reps = int(np.ceil(total_steps / N_orbit))
    P_timeline = np.tile(P_inst, reps)[:total_steps]
    t_timeline = np.arange(total_steps) * dt_step

    soc_wh = start_soc / 100.0 * batt_Wh
    soc_series = np.empty(total_steps)
    for k in range(total_steps):
        gen = P_timeline[k]
        load = cons_W
        if gen >= load:
            P_surplus = gen - load
            if P_chg_max is not None:
                P_surplus = min(P_surplus, P_chg_max)
            dE_Wh = (P_surplus * eta_chg) * dt_step / 3600.0
            soc_wh = min(soc_wh + dE_Wh, batt_Wh)
        else:
            P_deficit = load - gen
            dE_Wh = (P_deficit / eta_dis) * dt_step / 3600.0
            soc_wh = max(soc_wh - dE_Wh, 0.0)
        soc_series[k] = 100.0 * soc_wh / batt_Wh

    max_pts = 5000
    if total_steps > max_pts:
        idx = np.linspace(0, total_steps - 1, max_pts).astype(int)
        ts_plot = t_timeline[idx] / 3600.0
        soc_plot = soc_series[idx]
    else:
        ts_plot = t_timeline / 3600.0
        soc_plot = soc_series

    eod_idx = (np.arange(1, mission_days + 1) * int(np.floor(DAY_SEC / dt_step))).clip(
        0, total_steps - 1
    )
    eod_soc = soc_series[eod_idx]
    df_soc_daily = pd.DataFrame({"Day": np.arange(1, len(eod_idx) + 1), "SoC (%)": eod_soc})

    P_avg = float(P_inst.mean())
    daily_gen_Wh = P_avg * DAY_SEC / 3600.0
    daily_cons_Wh = cons_W * 24.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg generation (W)", f"{P_avg:.3f}")
    c2.metric("Daily generation (Wh)", f"{daily_gen_Wh:.1f}")
    c3.metric("Daily consumption (Wh)", f"{daily_cons_Wh:.1f}")

    st.plotly_chart(
        px.line(
            x=ts_plot,
            y=soc_plot,
            labels={"x": "Mission time (hours)", "y": "SoC (%)"},
            title="Battery SoC (mission time)",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(
            df_soc_daily,
            x="Day",
            y="SoC (%)",
            markers=True,
            title="Battery SoC — end of each day",
        ),
        use_container_width=True,
    )

    if len(eod_soc) and eod_soc[-1] < 20.0:
        st.warning(
            "Projected final SoC < 20% — consider more panel area/efficiency, better pointing, or lower load."
        )
    elif len(eod_soc):
        st.success("Battery SoC projection looks acceptable.")


# --- TAB 3: THERMAL ---
with tab_thermal:
    st.subheader("Radiative Thermal Equilibrium (current β)")

    # --- Inputs and basic 1-node result ---
    A_abs = st.number_input(
        "Absorbing area A_abs (m²)",
        0.001,
        2.0,
        sim.A_panel,
        0.001,
    )
    A_rad = st.number_input(
        "Radiating area A_rad (m²)",
        0.005,
        2.0,
        6.0 * sim.A_panel,
        0.005,
    )
    Q_int = st.number_input(
        "Internal dissipation Q_internal (W)",
        0.0,
        50.0,
        0.0,
        0.1,
    )

    # Keep Q_int around for Advanced thermal envelope section
    Q_int_total = Q_int

    T_c, Qs, Qa, Qir, Qin, Qtot = sim.thermal_equilibrium(
        A_abs=A_abs,
        A_rad=A_rad,
        Q_internal_W=Q_int,
    )

    # Shell temp only in this tab (no interior in 1-node model)
    st.metric("Shell equilibrium temp (°C)", f"{T_c:.2f}")

    dfQ = pd.DataFrame(
        [
            {
                "Solar_avg_W": Qs,
                "Albedo_W": Qa,
                "Earth_IR_W": Qir,
                "Internal_W": Qin,
                "Total_abs_W": Qtot,
            }
        ]
    )

    # --- Visuals: CubeSat rectangle + compact gauge ---
    cube_col, gauge_col = st.columns([1, 1.4])

    # Decide CubeSat color based on shell temp
    if T_c <= -10:
        cube_color = "#0ea5e9"
        cube_label = "Cold regime"
    elif T_c < 40:
        cube_color = "#22c55e"
        cube_label = "Nominal regime"
    else:
        cube_color = "#ef4444"
        cube_label = "Hot regime"

    # CubeSat rectangle
    with cube_col:
        st.markdown("### CubeSat body view")
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center;">
              <div style="
                  width:110px;
                  height:170px;
                  border-radius:20px;
                  border:2px solid #0f172a;
                  background:{cube_color};
                  box-shadow:0 10px 18px rgba(15,23,42,0.35);
              "></div>
              <div style="margin-top:0.5rem; font-size:0.85rem; color:#6b7280; text-align:center;">
                CubeSat shell (thermal indicator)<br>
                <span style="font-weight:600;">{cube_label}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gauge (shell temperature)
    with gauge_col:
        st.markdown("### Thermal gauge")

        temp_min = -40.0
        temp_max = 80.0

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=T_c,
                number={"suffix": " °C", "font": {"size": 28}},
                title={"text": "Shell Equilibrium Temp", "font": {"size": 16}, "align": "center"},
                gauge={
                    "axis": {"range": [temp_min, temp_max], "tickwidth": 1},
                    "bar": {"color": "#f97316"},
                    "steps": [
                        {"range": [temp_min, -10], "color": "#0ea5e9"},
                        {"range": [-10, 40], "color": "#22c55e"},
                        {"range": [40, temp_max], "color": "#ef4444"},
                    ],
                    "threshold": {
                        "line": {"color": "#111827", "width": 3},
                        "thickness": 0.75,
                        "value": T_c,
                    },
                },
            )
        )

        fig_gauge.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )

        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(
            "Blue = cold, green = nominal, red = hot. Adjust α/ε, radiating area, or internal power to move the temperature."
        )

    # --- Heat breakdown ---
    st.markdown("### Heat balance breakdown")
    st.dataframe(dfQ, use_container_width=True)
    st.plotly_chart(
        px.bar(
            dfQ.melt(var_name="Component", value_name="W"),
            x="Component",
            y="W",
            title="Absorbed power components (current β)",
        ),
        use_container_width=True,
    )


# --- TAB 4: DRAG & DEBRIS ---
with tab_drag:
    st.subheader("Altitude Decay from Drag (simple mission view)")
    A_drag = st.number_input("Reference drag area (m²)", 0.001, 2.0, sim.A_panel, 0.001)
    alt_series = sim.drag_decay_days(mission_days, A_drag=A_drag)
    df_alt = pd.DataFrame({"Day": np.arange(1, mission_days + 1), "Altitude (km)": alt_series})
    st.plotly_chart(
        px.line(
            df_alt,
            x="Day",
            y="Altitude (km)",
            markers=True,
            title="Altitude decay over mission",
        ),
        use_container_width=True,
    )


# --- TAB 5: VERIFICATION (Heritage) ---
with tab_verify:
    st.subheader("Verification vs Heritage Missions")

    st.markdown(
        """
        CATSIM is anchored to heritage CubeSat missions by matching key orbital and power
        characteristics, especially GeneSat-1 (NASA Ames / Stanford). This tab shows how
        the internal model calibrates to GeneSat-1 under canonical conditions.
        """
    )

    # --- Heritage reference values ---
    heritage_rows = [
        {
            "Mission": "GeneSat-1",
            "Mean altitude (km)": GENESAT_REF["alt_km"],
            "Inclination (deg)": GENESAT_REF["inc_deg"],
            "Orbital period (min)": GENESAT_REF["period_min"],
            "OAP (W)": GENESAT_REF["oap_W"],
            "Notes": "NASA/Ames 3U biology CubeSat; LEO, ~416.5 km, ~40°",
        },
        {
            "Mission": "PharmaSat",
            "Mean altitude (km)": PHARMASAT_REF["alt_km"],
            "Inclination (deg)": PHARMASAT_REF["inc_deg"],
            "Orbital period (min)": PHARMASAT_REF["period_min"],
            "OAP (W)": PHARMASAT_REF["oap_W"],
            "Notes": "Ames biology mission; similar bus to GeneSat-1",
        },
        {
            "Mission": "O/OREOS",
            "Mean altitude (km)": OOREOS_REF["alt_km"],
            "Inclination (deg)": OOREOS_REF["inc_deg"],
            "Orbital period (min)": OOREOS_REF["period_min"],
            "OAP (W)": OOREOS_REF["oap_W"],
            "Notes": "Ames 3U; higher-altitude comparison point",
        },
    ]
    df_herit = pd.DataFrame(heritage_rows)
    st.markdown("### Heritage reference values")
    st.dataframe(df_herit, use_container_width=True)
    st.caption(
        "OAP = orbit-average power. Values above are representative; refer to mission ODAR/papers for final numbers."
    )

    # Helper for relative error
    def rel_err(val, ref):
        if ref == 0:
            return 0.0
        return 100.0 * (val - ref) / ref

    # --- Model calibration at GeneSat-1 conditions ---
    st.markdown("### Model calibration at GeneSat-1 conditions")

    # Dedicated internal GeneSat-1 sim at canonical orbit
    sim_g = CubeSatSim(
        altitude_km=GENESAT_REF["alt_km"],
        incl_deg=GENESAT_REF["inc_deg"],
        mass_kg=GENESAT_REF["mass_kg"],
        Cd=GENESAT_DEFAULTS["Cd"],
        panel_area_m2=GENESAT_DEFAULTS["panel_area_m2"],
        panel_eff=GENESAT_DEFAULTS["panel_eff"],
        absorptivity=GENESAT_DEFAULTS["absorptivity"],
        emissivity=GENESAT_DEFAULTS["emissivity"],
        beta_deg=0.0,  # canonical beta for calibration
    )
    sim_g.set_beta(0.0)

    # Raw OAP from physics model
    oap_g_raw = sim_g.avg_power("body-spin", sim_g.A_panel, sim_g.eta)

    # Calibration factor to match GeneSat reference OAP (pre-derate)
    cal_g = GENESAT_REF["oap_W"] / oap_g_raw if oap_g_raw > 1e-9 else 1.0
    oap_g_cal = oap_g_raw * cal_g  # this will match 4.5 W within floating-point
    period_g_min = sim_g.T_orbit / 60.0

    df_cal = pd.DataFrame(
        [
            {
                "Quantity": "Mean altitude (km)",
                "CATSIM (GeneSat-1 mode)": sim_g.alt_km,
                "GeneSat-1 reference": GENESAT_REF["alt_km"],
                "Rel. diff vs ref (%)": rel_err(sim_g.alt_km, GENESAT_REF["alt_km"]),
            },
            {
                "Quantity": "Orbital period (min)",
                "CATSIM (GeneSat-1 mode)": period_g_min,
                "GeneSat-1 reference": GENESAT_REF["period_min"],
                "Rel. diff vs ref (%)": rel_err(period_g_min, GENESAT_REF["period_min"]),
            },
            {
                "Quantity": "Orbit-avg power (W, calibrated — pre-derate)",
                "CATSIM (GeneSat-1 mode)": oap_g_cal,
                "GeneSat-1 reference": GENESAT_REF["oap_W"],
                "Rel. diff vs ref (%)": rel_err(oap_g_cal, GENESAT_REF["oap_W"]),
            },
        ]
    )

    st.dataframe(df_cal, use_container_width=True)
    st.caption(
        "In this calibration panel, CATSIM uses a dedicated internal GeneSat-1 run and a gain factor "
        "`cal_g` such that the **pre-derate orbit-average power is matched to 4.5 W within ~1%**. "
        "Altitude and orbital period also agree with the published GeneSat-1 parameters within ~1%."
    )


# --- TAB 6: ADVANCED (Pro-only) ---
with tab_adv:
    st.markdown("## Advanced Analysis (Pro)")
    st.caption("Multi-β, multi-orbit, thermal envelope, debris compliance, and lifetime visualizations.")

    if plan_effective != "pro":
        st.info(
            "🔒 Advanced Analysis is available on the Pro plan ($19.99/mo).\n\n"
            "Your 30-day trial provides Standard features only."
        )
        st.markdown(
            "- Multi-β power curves for all attitudes\n"
            "- Multi-orbit SoC timeline\n"
            "- Thermal envelope vs β\n"
            "- Ground-track density map\n"
            "- NASA Orbital Debris Compliance (25-year rule)\n"
            "- Extended orbit lifetime curve"
        )
    else:
        # 1. Multi-β power curves
        st.markdown("### 1. Orbit-average Power vs β for Multiple Attitudes")
        betas_adv = np.linspace(-80, 80, 97)
        att_list = ["body-spin", "sun-tracking", "nadir-pointing"]
        rows = []
        for att in att_list:
            for b in betas_adv:
                oap_val = sim.avg_power_at_beta(att, b, sim.A_panel, sim.eta)
                rows.append(
                    {
                        "beta_deg": b,
                        "OAP_W": oap_val * cal_factor * elec_derate,
                        "Attitude": att,
                    }
                )
        df_multi_beta = pd.DataFrame(rows)
        fig_multi_beta = px.line(
            df_multi_beta,
            x="beta_deg",
            y="OAP_W",
            color="Attitude",
            labels={"beta_deg": "β (deg)", "OAP_W": "Orbit-average Power (W)"},
            title="Orbit-average Power vs β for Multiple Attitudes",
        )
        st.plotly_chart(fig_multi_beta, use_container_width=True)

        # 2. Multi-orbit SoC
        st.markdown("### 2. Multi-orbit Battery SoC (mission timeline)")
        N_orbit_adv = 720
        t_orb_adv, u_adv, x_km_adv, y_km_adv, z_km_adv = sim.orbit_eci(N=N_orbit_adv)
        dt_adv = float(t_orb_adv[1] - t_orb_adv[0])
        P_inst_adv, _, _ = sim.instantaneous_power(
            attitude, t_orb_adv, x_km_adv, y_km_adv, z_km_adv, sim.A_panel, sim.eta
        )
        P_inst_adv = P_inst_adv * cal_factor * elec_derate

        total_steps_adv = int(np.ceil((mission_days * DAY_SEC) / dt_adv))
        reps_adv = int(np.ceil(total_steps_adv / N_orbit_adv))
        P_timeline_adv = np.tile(P_inst_adv, reps_adv)[:total_steps_adv]
        t_timeline_adv = np.arange(total_steps_adv) * dt_adv

        soc_wh_adv = start_soc / 100.0 * batt_Wh
        soc_series_adv = np.empty(total_steps_adv)
        for k in range(total_steps_adv):
            gen = P_timeline_adv[k]
            load = cons_W
            if gen >= load:
                P_surplus = gen - load
                if "P_chg_max" in locals() and P_chg_max is not None:
                    P_surplus = min(P_surplus, P_chg_max)
                dE_Wh = (P_surplus * eta_chg) * dt_adv / 3600.0
                soc_wh_adv = min(soc_wh_adv + dE_Wh, batt_Wh)
            else:
                P_deficit = load - gen
                dE_Wh = (P_deficit / eta_dis) * dt_adv / 3600.0
                soc_wh_adv = max(soc_wh_adv - dE_Wh, 0.0)
            soc_series_adv[k] = 100.0 * soc_wh_adv / batt_Wh

        ts_plot_adv = t_timeline_adv / 3600.0
        st.plotly_chart(
            px.line(
                x=ts_plot_adv,
                y=soc_series_adv,
                labels={"x": "Mission time (hours)", "y": "SoC (%)"},
                title="Multi-orbit Battery SoC — mission timeline",
            ),
            use_container_width=True,
        )

        # 3. Thermal envelope vs β
        st.markdown("### 3. Thermal Envelope vs β")
        betas_th = np.linspace(-80, 80, 65)
        temps_eq = []
        for b in betas_th:
            T_eq_b, *_ = sim.thermal_equilibrium(
                A_abs=A_abs, A_rad=A_rad, Q_internal_W=Q_int, beta_for_thermal=b
            )
            temps_eq.append(T_eq_b)
        df_temp = pd.DataFrame({"beta_deg": betas_th, "T_eq_C": temps_eq})
        fig_temp = px.line(
            df_temp,
            x="beta_deg",
            y="T_eq_C",
            labels={"beta_deg": "β (deg)", "T_eq_C": "Equilibrium Temp (°C)"},
            title="Equilibrium Temperature vs β",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        c_min, c_max = st.columns(2)
        c_min.metric("Min T_eq over β (°C)", f"{df_temp['T_eq_C'].min():.1f}")
        c_max.metric("Max T_eq over β (°C)", f"{df_temp['T_eq_C'].max():.1f}")

        # 3b. Two-node Thermal Analysis (shell + interior)
        st.markdown("### 3b. Two-node Thermal Analysis (Shell + Interior)")

        A_abs_2 = st.number_input(
            "Absorbing area A_abs (m²) — 2-node thermal",
            0.001,
            2.0,
            sim.A_panel,
            0.001,
            key="th2_A_abs",
        )
        A_rad_2 = st.number_input(
            "Radiating area A_rad (m²) — 2-node thermal",
            0.005,
            2.0,
            6.0 * sim.A_panel,
            0.005,
            key="th2_A_rad",
        )
        Q_int_total_2 = st.number_input(
            "Total internal dissipation Q_internal (W) — 2-node thermal",
            0.0,
            50.0,
            5.0,
            0.1,
            key="th2_Q",
        )
        frac_int_2 = st.slider(
            "Fraction of internal power in interior node",
            0.0,
            1.0,
            0.8,
            0.05,
            key="th2_frac",
        )
        k_cond_2 = st.slider(
            "Conduction between shell and interior (W/K)",
            0.1,
            10.0,
            1.0,
            0.1,
            key="th2_kcond",
        )

        Q_int_int_2 = Q_int_total_2 * frac_int_2
        Q_int_shell_2 = Q_int_total_2 * (1.0 - frac_int_2)

        T_shell_C_2, T_int_C_2, thermo_diag_2 = sim.thermal_equilibrium_2node(
            A_abs_shell=A_abs_2,
            A_rad_shell=A_rad_2,
            Q_int_shell_W=Q_int_shell_2,
            Q_int_interior_W=Q_int_int_2,
            k_cond_W_per_K=k_cond_2,
            beta_for_thermal=None,  # use current sim.beta_deg
            T_shell_init_C=0.0,
            T_int_init_C=0.0,
        )

        c_ts2, c_ti2 = st.columns(2)
        c_ts2.metric("Shell equilibrium temp (°C)", f"{T_shell_C_2:.2f}")
        c_ti2.metric("Interior equilibrium temp (°C)", f"{T_int_C_2:.2f}")

        dfQ2 = pd.DataFrame(
            [
                {
                    "Solar_shell_W": thermo_diag_2["Q_solar_shell_W"],
                    "Albedo_shell_W": thermo_diag_2["Q_albedo_shell_W"],
                    "Earth_IR_shell_W": thermo_diag_2["Q_ir_shell_W"],
                    "Internal_shell_W": thermo_diag_2["Q_int_shell_W"],
                    "Internal_interior_W": thermo_diag_2["Q_int_interior_W"],
                    "Shell_rad_to_space_W": thermo_diag_2["Q_rad_shell_W"],
                    "Cond_from_interior_W": thermo_diag_2["Q_cond_shell_W"],
                }
            ]
        )

        st.dataframe(dfQ2, use_container_width=True)
        st.plotly_chart(
            px.bar(
                dfQ2.melt(var_name="Component", value_name="W"),
                x="Component",
                y="W",
                title="Two-node thermal power balance (shell + interior)",
            ),
            use_container_width=True,
        )

        # 4. Ground-track density map
        st.markdown("### 4. Ground-track Density Map")
        dens_days = st.slider(
            "Days for density map", 1, min(60, mission_days), min(7, mission_days)
        )
        num_orbits_dens = max(1, int(np.ceil(dens_days * DAY_SEC / sim.T_orbit)))

        t_long, u_long, x_long, y_long, z_long = sim.long_orbit_eci(
            num_orbits=num_orbits_dens, N_per_orbit=360
        )
        lon_long, lat_long = sim.ground_track_from_eci(t_long, x_long, y_long, z_long)

        df_dens = pd.DataFrame({"lon": lon_long, "lat": lat_long})
        fig_dens = px.density_heatmap(
            df_dens,
            x="lon",
            y="lat",
            nbinsx=72,
            nbinsy=36,
            labels={"lon": "Longitude (deg)", "lat": "Latitude (deg)"},
            title=f"Ground-track Density over ~{dens_days} days",
        )
        st.plotly_chart(fig_dens, use_container_width=True)

        # 5. Orbit lifetime curve (drag decay)
        st.markdown("### 5. Orbit Lifetime Curve (Drag Decay)")
        lifetime_days = st.slider(
            "Lifetime analysis (days)", mission_days, 3650, max(mission_days, 365)
        )
        A_drag_adv = st.number_input(
            "Drag area A_drag (m²) for lifetime", 0.001, 2.0, sim.A_panel, 0.001
        )

        alt_series_adv = sim.drag_decay_days(lifetime_days, A_drag=A_drag_adv)
        df_alt_adv = pd.DataFrame(
            {"Day": np.arange(1, lifetime_days + 1), "Altitude (km)": alt_series_adv}
        )
        fig_alt_adv = px.line(
            df_alt_adv,
            x="Day",
            y="Altitude (km)",
            title="Orbit Lifetime Curve (Altitude vs Day)",
        )
        st.plotly_chart(fig_alt_adv, use_container_width=True)

        if len(alt_series_adv):
            st.metric("Final altitude (km)", f"{alt_series_adv[-1]:.1f}")

        # 6. NASA Orbital Debris Compliance (25-year rule)
        st.markdown("### 6. NASA Orbital Debris Compliance (25-year rule)")

        solar_choice = st.radio(
            "Solar activity assumption (for debris lifetime)",
            ["Low", "Nominal", "High"],
            index=1,
            horizontal=True,
        )
        solar_scale = {"Low": 0.7, "Nominal": 1.0, "High": 1.4}[solar_choice]

        disposal_mode = st.selectbox(
            "End-of-mission disposal strategy",
            [
                "Natural decay from mission orbit",
                "Propulsive deorbit to lower circular orbit",
                "Drag augmentation device deployment",
            ],
        )

        sma_km = (R_E + altitude_km * 1000.0) / 1000.0
        ecc = 0.0
        mission_life_years = mission_days / 365.0

        od_result = nasa_debris_compliance_check(
            sma_km=sma_km,
            ecc=ecc,
            mass_kg=mass_kg,
            cross_section_m2=panel_area,
            cd=Cd,
            solar_activity_scale=solar_scale,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Perigee altitude", f"{od_result['perigee_alt_km']:.0f} km")
        c2.metric("Est. post-mission lifetime", f"{od_result['lifetime_years']:.1f} years")
        c3.metric("Compliance status", f"{od_result['emoji']} {od_result['status']}")

        st.markdown(f"**Interpretation:** {od_result['note']}")

        odar_text = generate_odar_summary_text(
            mission_name="CATSIM Mission",
            sma_km=sma_km,
            ecc=ecc,
            inc_deg=incl_deg,
            mass_kg=mass_kg,
            cross_section_m2=panel_area,
            mission_life_years=mission_life_years,
            disposal_mode=disposal_mode,
            compliance_result=od_result,
        )

        st.markdown("#### ODAR Narrative (copy into your documentation)")
        st.text_area("ODAR Summary", odar_text, height=280)


# --- TAB 7: SAVE/LOAD & EXPORT (Pro-only) ---
with tab_io:
    st.subheader("Save/Load & Export (Pro)")
    st.markdown("⬇️ **Save / Load Missions & Export Data**")

    if plan_effective != "pro":
        st.info(
            "🔒 Save/Load & Export are available on the Pro plan ($19.99/mo).\n\n"
            "Your 30-day trial provides Standard features only."
        )
        st.write("- Save mission parameters to JSON")
        st.write("- Export Orbit CSV and Power CSV")
        st.write("")
        st.write("Upgrade to Pro in the sidebar to unlock.")
    else:
        mission_params = {
            "altitude_km": altitude_km,
            "incl_deg": incl_deg,
            "mass_kg": mass_kg,
            "Cd": Cd,
            "panel_area_m2": panel_area,
            "panel_eff": panel_eff,
            "absorptivity": absorp,
            "emissivity": emiss,
            "attitude": attitude,
            "auto_cal": auto_cal,
            "elec_derate": elec_derate,
            "beta_deg": beta_deg,
        }
        json_bytes = json.dumps(mission_params, indent=2).encode("utf-8")
        st.download_button(
            "Download Mission JSON",
            data=json_bytes,
            file_name="mission.json",
            mime="application/json",
        )

        t_orb2, u_orb, x_orb, y_orb, z_orb = sim.orbit_eci(N=720)
        P_orb, _, _ = sim.instantaneous_power(
            attitude, t_orb2, x_orb, y_orb, z_orb, sim.A_panel, sim.eta
        )
        P_orb = P_orb * cal_factor * elec_derate

        df_orbit = pd.DataFrame(
            {
                "t_sec": t_orb2,
                "x_eci_km": x_orb,
                "y_eci_km": y_orb,
                "z_eci_km": z_orb,
            }
        )
        buf_orbit = io.StringIO()
        df_orbit.to_csv(buf_orbit, index=False)
        st.download_button(
            "Export One-Orbit ECI CSV",
            buf_orbit.getvalue(),
            file_name="orbit_one_orbit.csv",
            mime="text/csv",
        )

        df_power_orbit = pd.DataFrame({"t_sec": t_orb2, "power_W": P_orb})
        buf_porb = io.StringIO()
        df_power_orbit.to_csv(buf_porb, index=False)
        st.download_button(
            "Export One-Orbit Power CSV",
            buf_porb.getvalue(),
            file_name="power_one_orbit.csv",
            mime="text/csv",
        )

# =========================
# Footer
# =========================
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 0.9em;'>
    © 2025 Cleveland Aerospace Technology Services — Davidsonville, MD<br>
    <a href="mailto:press@clevelandaerospace.com">press@clevelandaerospace.com</a> •
    <a href="mailto:support@clevelandaerospace.com">support@clevelandaerospace.com</a>
    </p>
    """,
    unsafe_allow_html=True,
)
