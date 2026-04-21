"""
database.py - SQLite Database Connection and Operations
Handles all database interactions using SQLite (no server required).
Database file: retail_analytics.db (auto-created on first run)
"""

import sqlite3
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Database Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "retail_analytics.db")


def get_connection():
    """
    Returns a SQLite connection with row_factory for dict-like access.
    Creates the database file automatically if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Allows dict-style column access
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent performance
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> bool:
    """
    Creates all required tables if they don't already exist.
    Called once on application startup.

    Tables:
        users         – stores registered user accounts
        uploaded_files – logs file upload history per user
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # ── users table ──────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    NOT NULL UNIQUE,
                email        TEXT    NOT NULL UNIQUE,
                password     TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                last_login   TEXT
            )
        """)

        # ── uploaded_files table ─────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                filename     TEXT    NOT NULL,
                row_count    INTEGER NOT NULL,
                uploaded_at  TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[DB] init_db error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# User Operations
# ─────────────────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, hashed_password: str) -> dict:
    """
    Inserts a new user into the users table.

    Returns:
        {'success': True, 'user_id': int}  on success
        {'success': False, 'error': str}   on failure
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, email, password, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username.strip(), email.strip(), hashed_password,
             datetime.utcnow().isoformat())
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "user_id": user_id}

    except sqlite3.IntegrityError as e:
        # Triggered when username or email already exists (UNIQUE constraint)
        if "username" in str(e):
            return {"success": False, "error": "Username already taken."}
        elif "email" in str(e):
            return {"success": False, "error": "Email already registered."}
        return {"success": False, "error": "User already exists."}

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_user_by_username(username: str) -> dict | None:
    """
    Fetches a user row by username.
    Returns a dict or None if not found.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def update_last_login(username: str):
    """Updates the last_login timestamp for a user."""
    try:
        conn = get_connection()
        conn.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow().isoformat(), username.strip())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# File Upload Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_upload(user_id: int, filename: str, row_count: int):
    """
    Inserts a record into uploaded_files to track upload history.
    """
    try:
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO uploaded_files (user_id, filename, row_count, uploaded_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, filename, row_count, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_user_uploads(user_id: int) -> list:
    """
    Returns all upload history for a given user, newest first.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT filename, row_count, uploaded_at
            FROM uploaded_files
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            """,
            (user_id,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Admin / Debug Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users() -> list:
    """Returns all registered users (for admin/debug purposes)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, created_at, last_login FROM users")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_db_stats() -> dict:
    """Returns basic database statistics."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM users")
        user_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) as cnt FROM uploaded_files")
        upload_count = cursor.fetchone()["cnt"]
        conn.close()
        return {
            "total_users": user_count,
            "total_uploads": upload_count,
            "db_path": os.path.abspath(DB_PATH)
        }
    except Exception:
        return {"total_users": 0, "total_uploads": 0, "db_path": DB_PATH}
