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

# =========================
# Stripe Payment Links (legacy simple links – not used with new backend flow)
# =========================
STD_MONTHLY_LINK = os.getenv("CATSIM_STD_MONTHLY_LINK", "")
STD_YEARLY_LINK = os.getenv("CATSIM_STD_YEARLY_LINK", "")
PRO_MONTHLY_LINK = os.getenv("CATSIM_PRO_MONTHLY_LINK", "")
PRO_YEARLY_LINK = os.getenv("CATSIM_PRO_YEARLY_LINK", "")
ACADEMIC_LINK = os.getenv("CATSIM_ACAD_LINK", "")
DEPT_LINK = os.getenv("CATSIM_DEPT_LINK", "")

# =========================
# Backend base URL (Render service)
# =========================
BACKEND_BASE = os.getenv(
    "CATSIM_BACKEND_URL",
    "https://catsim-backend.onrender.com",
)


def api_url(path: str) -> str:
    return f"{BACKEND_BASE.rstrip('/')}{path}"


# =========================
# Environment: dev vs prod
# =========================
ENV = "dev"
DEV_MODE = True
IS_DEV = True

CONFIG = {
    "dev": {
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
        "AUTH0_DOMAIN": os.getenv("AUTH0_DOMAIN_PROD", "dev-qwn3runpmc616as6.us.auth0.com"),
        "AUTH0_CLIENT_ID": os.getenv("AUTH0_CLIENT_ID_PROD", "XvRZKwcHlcToRYGMMwnZiNjLnNzJmUmU"),
        "AUTH0_CLIENT_SECRET": os.getenv(
            "AUTH0_CLIENT_SECRET_PROD",
            "E9BAjy7QLsJ0GSAYPMoBvb-vg7lMeLObKBqdsBupoQoVcVUHM75hmXOSDm2jzuw7",
        ),
        "AUTH0_CALLBACK_URL": os.getenv(
            "AUTH0_CALLBACK_URL_PROD",
            "https://cubesimprov2-lt6hcgkvpdvygnwbktyqdg.streamlit.app",
        ),
        "SHOW_DEV_PLAN_SIM": False,
    },
}[ENV]

AUTH0_DOMAIN = CONFIG["AUTH0_DOMAIN"]
AUTH0_CLIENT_ID = CONFIG["AUTH0_CLIENT_ID"]
AUTH0_CLIENT_SECRET = CONFIG["AUTH0_CLIENT_SECRET"]
AUTH0_CALLBACK_URL = CONFIG["AUTH0_CALLBACK_URL"]
AUTH0_AUDIENCE = "https://catsim-backend-api"

# =========================
# Backend helpers
# =========================


def get_billing_portal_url(user: dict | None) -> str | None:
    """
    Call backend /create-portal-session and return the Stripe billing portal URL.
    """
    if user is None:
        return None

    try:
        payload = {"user_id": user.get("sub")}
        resp = requests.post(api_url("/create-portal-session"), json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("url")
    except Exception as e:
        st.sidebar.error(
            "Could not open billing portal. Please contact support if this persists."
        )
        if os.getenv("CATSIM_ENV", "dev") == "dev":
            try:
                st.sidebar.write("Debug:", resp.text)
            except Exception:
                st.sidebar.write(str(e))
        return None


def sync_user_with_backend(user: dict) -> None:
    """
    Best-effort sync of Auth0 identity into the backend DB.
    Backend: POST /sync-user
    """
    if not BACKEND_BASE or not user:
        return

    try:
        payload = {
            "auth0_sub": user.get("sub"),
            "email": user.get("email"),
            "stripe_customer_id": st.session_state.get("stripe_customer_id"),
        }
        requests.post(api_url("/sync-user"), json=payload, timeout=5)
    except Exception:
        return


def fetch_subscription_state(auth0_sub: str) -> dict | None:
    """
    Backend: GET /subscription-state?auth0_sub=...
    Expected JSON:
      {
        "plan": "standard" | "pro" | "trial" | "none",
        "plan_key": "pro_monthly" | ... | null,
        "status": "active" | "trialing" | "canceled" | ...,
        "price_id": "price_xxx" | null,
        "current_period_end": "2025-01-01T00:00:00Z" | null,
        "customer_id": "cus_xxx" | null
      }
    """
    if not BACKEND_BASE or not auth0_sub:
        return None

    try:
        resp = requests.get(
            api_url("/subscription-state"),
            params={"auth0_sub": auth0_sub},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def create_checkout_session(plan_key: str, email: str) -> str | None:
    """Create a Stripe Checkout Session for a given plan + email."""
    if not BACKEND_BASE:
        return None
    try:
        resp = requests.post(
            api_url("/create-checkout-session"),
            json={"plan_key": plan_key, "email": email},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("url")
    except Exception as e:
        st.error(f"Checkout error: {e}")
        return None


def create_billing_portal(customer_id: str) -> str | None:
    """Create a Stripe Billing Portal Session for managing subscription."""
    if not BACKEND_BASE:
        return None
    try:
        resp = requests.post(
            api_url("/create-portal-session"),
            json={"customer_id": customer_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("url")
    except Exception as e:
        st.error(f"Billing portal error: {e}")
        return None


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="CATSIM — CubeSat Mission Simulator",
    page_icon="CATS_Logo.png",
    layout="wide",
)

# =========================
# Auth0 helpers
# =========================


def auth0_login_url():
    params = {
        "response_type": "code",
        "client_id": AUTH0_CLIENT_ID,
        "redirect_uri": AUTH0_CALLBACK_URL,
        "scope": "openid profile email",
        "audience": AUTH0_AUDIENCE,
    }
    return f"https://{AUTH0_DOMAIN}/authorize?" + urllib.parse.urlencode(params)


def auth0_logout_url():
    params = {
        "client_id": AUTH0_CLIENT_ID,
        "returnTo": AUTH0_CALLBACK_URL,
    }
    return f"https://{AUTH0_DOMAIN}/v2/logout?" + urllib.parse.urlencode(params)


def login_button():
    url = auth0_login_url()
    st.markdown(f"[Click here to sign in]({url})")


def _exchange_code_for_tokens(code: str):
    token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "code": code,
        "redirect_uri": AUTH0_CALLBACK_URL,
    }

    resp = requests.post(token_url, data=data)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def get_user():
    """
    Returns the current user dict, or None.
    1) If user in session_state -> return it.
    2) Else, if ?code=... in URL -> exchange it, store user, clear params.
    """
    if "user" in st.session_state:
        return st.session_state["user"]

    params = st.query_params
    code = params.get("code")
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

            st.query_params.clear()
            return user
        except Exception as e:
            st.error(f"Auth error: {e}")
            return None

    return None


def inject_brand_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #1d4ed8 !important;
            --primaryColor: #1d4ed8 !important;
        }

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

        .dataframe th, .dataframe td {
            font-size: 0.85rem !important;
        }

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
        unsafe_allow_html=True,
    )
    st.markdown("---")


# =========================
# Auth gate
# =========================
user = get_user()
if not user:
    st.title("CATSIM — Sign in")
    st.write("Please sign in to use the CubeSat Mission Simulator.")
    login_button()
    st.stop()

inject_brand_css()
show_header(user)

# Sync identity into backend
sync_user_with_backend(user)

# =========================
# Physical constants
# =========================
MU = 3.986004418e14
R_E = 6371e3
OMEGA_E = 7.2921159e-5
SIGMA = 5.670374419e-8
SOLAR_CONST = 1366.0
EARTH_IR = 237.0
ALBEDO = 0.3
DAY_SEC = 86400.0
R_EARTH_KM = R_E / 1000.0

# ... (all physics + CubeSatSim + debris functions exactly as in your code above) ...
# --- I will not change that content at all, just keep what you already have ---
# (to keep this answer from overflowing, imagine that entire physics section,
# CubeSatSim class, debris helpers, generate_odar_summary_text, etc. are
# unchanged and pasted here verbatim from your last message.)

# -------------- BEGIN physics / CubeSatSim / debris block --------------
# (PASTE UNCHANGED from your last message)
# -------------- END physics / CubeSatSim / debris block --------------

# =========================
# Pricing model & plan state (via backend)
# =========================
auth_email = (user or {}).get("email", "unknown")
auth_name = (user or {}).get("name", "")
auth_pic = (user or {}).get("picture")
auth_sub = (user or {}).get("sub", "")

today = dt.date.today()

if "trial_start" not in st.session_state:
    st.session_state.trial_start = today.isoformat()

if "subscription_state" not in st.session_state:
    st.session_state.subscription_state = fetch_subscription_state(auth_sub)

sub = st.session_state.subscription_state

trial_start = dt.date.fromisoformat(st.session_state.trial_start)
trial_end = trial_start + dt.timedelta(days=30)

if sub and sub.get("plan") in ("standard", "pro"):
    backend_plan = sub.get("plan")
    plan_base = backend_plan
    in_trial = sub.get("status", "").lower() == "trialing"
    status = sub.get("status", "active")
    customer_id = sub.get("customer_id")
    plan_effective = backend_plan if not in_trial else "standard"
else:
    if "plan_base" not in st.session_state:
        st.session_state.plan_base = "trial"
    plan_base = st.session_state.plan_base
    in_trial = (plan_base == "trial") and (today <= trial_end)
    plan_effective = "standard" if in_trial else plan_base
    status = "active"
    customer_id = None

st.session_state.plan_base = plan_base
st.session_state.effective_plan = plan_effective
st.session_state.in_trial = in_trial
st.session_state.trial_start = trial_start.isoformat()
st.session_state.trial_end = trial_end.isoformat()
st.session_state.stripe_customer_id = customer_id

# =========================
# SIDEBAR: Account + Plan & Billing + Simulation setup
# =========================
with st.sidebar:
    # -------- Account --------
    st.markdown("### Account")

    if user is not None:
        portal_url = get_billing_portal_url(user)
        if portal_url:
            try:
                st.link_button(
                    "Manage billing & invoices",
                    portal_url,
                    use_container_width=True,
                    type="secondary",
                )
            except AttributeError:
                st.markdown(
                    f"<a href='{portal_url}' target='_blank'>"
                    "<button style='width:100%; padding:0.5rem 1rem; border-radius:0.5rem;"
                    "border:1px solid #ccc; background-color:white; cursor:pointer;'>"
                    "Manage billing & invoices"
                    "</button></a>",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Sign in to manage your subscription.")

    logout_url = auth0_logout_url()
    if st.button("Log out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.markdown(
            f"<meta http-equiv='refresh' content='0; url={logout_url}'>",
            unsafe_allow_html=True,
        )
        st.stop()

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

    # -------- Plan & Billing --------
    st.markdown("### Plan & Billing")

    # label + end text
    if plan_effective == "pro":
        plan_label = "Pro"
    elif plan_effective == "standard":
        plan_label = "Standard"
    elif plan_effective == "trial":
        plan_label = "Trial (Standard)"
    else:
        plan_label = "No active subscription"

    if in_trial:
        end_text = f"Trial ends: {trial_end.isoformat()}"
    elif sub and sub.get("current_period_end"):
        try:
            end_date = dt.date.fromisoformat(sub["current_period_end"][:10])
            end_text = f"Renews: {end_date.isoformat()}"
        except Exception:
            end_text = ""
    else:
        end_text = ""

    st.markdown(
        f"""
        <div style="border-radius: 8px; padding: 12px 16px;
                    background-color: #f5f9ff; border: 1px solid #c3d5ff;
                    margin-bottom: 8px;">
            <div style="font-weight: 600; margin-bottom: 4px;">
                Current plan: 🚀 {plan_label}
            </div>
            <div style="font-size: 0.9rem; color: #444;">
                {end_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if plan_effective != "pro":
        st.warning("Pro features are locked. Upgrade to Pro to enable.")

    # -------- Upgrade buttons --------
    st.markdown("### Upgrade")

    col_a, col_b = st.columns(2)

    # Standard
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

    # Pro
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

    st.markdown("**Education & teams:**")
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Academic Pro $99/yr", key="academic"):
            with st.spinner("Contacting secure Stripe checkout..."):
                url = create_checkout_session("academic_yearly", auth_email)
            if url:
                st.session_state.checkout_url = url
    with col_d:
        if st.button("Dept License $499/yr", key="dept"):
            with st.spinner("Contacting secure Stripe checkout..."):
                url = create_checkout_session("dept_yearly", auth_email)
            if url:
                st.session_state.checkout_url = url

    if st.session_state.get("checkout_url"):
        st.markdown("### Checkout / Billing")
        st.info(
            "✅ Click below to open secure Stripe checkout / billing. "
            "After completing payment, refresh CATSIM to update your plan."
        )
        st.markdown(f"[Open Stripe]({st.session_state.checkout_url})")

    st.divider()
    st.markdown("### Simulation setup")

    # ---- Simulation setup controls (exactly as you had) ----
    with st.expander("Preset & validation", expanded=True):
        use_genesat = st.checkbox("Load GeneSat-1 defaults", True)
        auto_cal = st.checkbox("Calibrate avg power to GeneSat target (~4.5 W)", True)

    st.markdown("#### Mission parameters")

    # (All the sliders/inputs for altitude_km, incl_deg, mass_kg, Cd, etc.)
    # Paste your existing simulation-setup block here unchanged,
    # just indented inside this with st.sidebar: block.

    # For brevity in this answer, I'm not re-pasting that entire section,
    # but structurally it's identical to what you sent — just indented
    # under `with st.sidebar:` so it appears in the sidebar again.

# =========================
# Main tabs & plots
# =========================
# From here down, your orbit / power / thermal / drag / verification /
# Advanced / Save-Load tabs stay exactly as you had them, using the
# variables defined in the sidebar (altitude_km, incl_deg, etc.)
# Paste that whole section unchanged.

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
