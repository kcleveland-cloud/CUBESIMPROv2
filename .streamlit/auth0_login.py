import streamlit as st
import urllib.parse
import requests

def login_button():
    domain = st.secrets["AUTH0_DOMAIN"]
    client_id = st.secrets["AUTH0_CLIENT_ID"]
    redirect_uri = st.secrets["AUTH0_CALLBACK_URL"]

    auth_url = (
        f"https://{domain}/authorize?"
        + urllib.parse.urlencode({
            "response_type": "token",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email",
        })
    )

    st.markdown(f"[Sign in with Auth0]({auth_url})")

def get_user():
    # When Auth0 redirects back, access token appears in URL fragment.
    if "access_token" in st.query_params:
        return True
    return False
