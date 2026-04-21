"""
auth.py - Authentication Module
Handles user registration, login, and Streamlit session management.
Uses bcrypt for secure password hashing + SQLite for storage.
"""

import streamlit as st
import bcrypt
from database import create_user, get_user_by_username, update_last_login


# ─────────────────────────────────────────────────────────────────────────────
# Password Hashing
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a stored bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> dict:
    """
    Validates inputs, hashes password, and stores user in SQLite.
    Returns {'success': True} or {'success': False, 'error': '...'}
    """
    username = username.strip()
    email    = email.strip()

    if len(username) < 3:
        return {"success": False, "error": "Username must be at least 3 characters."}
    if "@" not in email or "." not in email:
        return {"success": False, "error": "Please enter a valid email address."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    hashed = hash_password(password)
    return create_user(username, email, hashed)


# ─────────────────────────────────────────────────────────────────────────────
# Login
# ─────────────────────────────────────────────────────────────────────────────

def login_user(username: str, password: str) -> dict:
    """
    Looks up user in SQLite and verifies password.
    Returns {'success': True, 'user': {...}} or error dict.
    """
    user = get_user_by_username(username.strip())
    if user is None:
        return {"success": False, "error": "No account found with that username."}

    if not verify_password(password, user["password"]):
        return {"success": False, "error": "Incorrect password. Please try again."}

    update_last_login(username)

    return {
        "success": True,
        "user": {
            "user_id":  user["id"],
            "username": user["username"],
            "email":    user["email"]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session State Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def get_current_user() -> dict | None:
    return st.session_state.get("current_user", None)


def logout():
    st.session_state["logged_in"]    = False
    st.session_state["current_user"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Auth Page UI
# ─────────────────────────────────────────────────────────────────────────────

def render_auth_page():
    """
    Full-page login / register UI rendered when user is not authenticated.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stTextInput > label { color: rgba(255,255,255,0.7) !important; font-size:0.85rem !important; }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; width: 100% !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important; font-size: 1rem !important;
        padding: 0.75rem !important; transition: all 0.3s !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(124,58,237,0.5) !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        # Logo
        st.markdown("""
        <div style='text-align:center; padding: 2.5rem 0 1.5rem;'>
            <div style='font-size:3rem;'>📊</div>
            <h1 style='font-size:2.2rem; font-weight:800;
                        background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
                RetailIQ
            </h1>
            <p style='color:rgba(255,255,255,0.4); font-size:0.85rem; margin:0.3rem 0 0;'>
                Integrated Retail Analytics Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            _login_form()

        with tab_register:
            _register_form()

        # DB notice
        st.markdown("""
        <p style='text-align:center; color:rgba(255,255,255,0.2); font-size:0.7rem; margin-top:1.5rem;'>
            🗄️ Powered by SQLite · No server required
        </p>
        """, unsafe_allow_html=True)


def _login_form():
    st.markdown("<br>", unsafe_allow_html=True)
    username = st.text_input("Username", placeholder="Enter your username", key="li_user")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Sign In →", key="btn_login"):
        if not username or not password:
            st.error("Please fill in all fields.")
        else:
            res = login_user(username, password)
            if res["success"]:
                st.session_state["logged_in"]    = True
                st.session_state["current_user"] = res["user"]
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error(f"❌ {res['error']}")


def _register_form():
    st.markdown("<br>", unsafe_allow_html=True)
    username = st.text_input("Username", placeholder="Choose a username (min 3 chars)", key="reg_user")
    email    = st.text_input("Email",    placeholder="Enter your email address",         key="reg_email")
    password = st.text_input("Password", type="password", placeholder="Create a strong password (min 6 chars)", key="reg_pass")
    confirm  = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="reg_conf")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Create Account →", key="btn_register"):
        if not all([username, email, password, confirm]):
            st.error("Please fill in all fields.")
        elif password != confirm:
            st.error("❌ Passwords do not match.")
        else:
            res = register_user(username, email, password)
            if res["success"]:
                st.success("✅ Account created! You can now sign in.")
            else:
                st.error(f"❌ {res['error']}")
