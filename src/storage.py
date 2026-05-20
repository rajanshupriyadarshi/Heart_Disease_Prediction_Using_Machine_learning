import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "app.db"
ADMIN_EMAIL = "admin@heartcare.local"
ADMIN_NAME = "System Admin"
ADMIN_PHONE = "0000000000"
ADMIN_PASSWORD = "Admin@123"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        100000,
    ).hex()
    return salt, hashed


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, hashed = hash_password(password, salt)
    return hashed == password_hash


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_data TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.commit()
    ensure_admin_user()


def ensure_admin_user() -> None:
    if get_user_by_email(ADMIN_EMAIL):
        return
    create_user(
        full_name=ADMIN_NAME,
        email=ADMIN_EMAIL,
        phone=ADMIN_PHONE,
        password=ADMIN_PASSWORD,
        role="admin",
    )


def create_user(full_name: str, email: str, phone: str, password: str, role: str = "user") -> int:
    salt, password_hash = hash_password(password)
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (full_name, email, phone, password_salt, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (full_name, email.lower(), phone, salt, password_hash, role, created_at),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_by_email(email: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower(),),
        ).fetchone()


def authenticate_user(email: str, password: str, role: str | None = None):
    user = get_user_by_email(email)
    if not user:
        return None
    if role and user["role"] != role:
        return None
    if not verify_password(password, user["password_salt"], user["password_hash"]):
        return None
    return user


def save_prediction(user_id: int, patient_data: dict, prediction: int, probability: float, label: str) -> None:
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (user_id, patient_data, prediction, probability, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, json.dumps(patient_data), prediction, probability, label, created_at),
        )
        conn.commit()


def get_user_predictions(user_id: int):
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, prediction, probability, label, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def get_all_users():
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, full_name, email, phone, role, created_at FROM users ORDER BY id DESC"
        ).fetchall()


def get_all_predictions():
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT p.id, u.full_name, u.email, p.prediction, p.probability, p.label, p.created_at
            FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC
            """
        ).fetchall()


def delete_user_and_data(user_id: int) -> bool:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not user or user["role"] == "admin":
            return False

        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True


def clear_user_history(user_id: int) -> bool:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not user or user["role"] == "admin":
            return False

        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
