import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("app_data.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            apellido TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS detecciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id INTEGER,
            emocion TEXT NOT NULL,
            confianza REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (persona_id) REFERENCES personas(id)
        )
        """)
        conn.commit()

def insert_persona(nombre, apellido, email, embedding_json):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO personas (nombre, apellido, email, embedding)
        VALUES (?, ?, ?, ?)
        """, (nombre, apellido, email, embedding_json))
        conn.commit()
        return cur.lastrowid

def get_all_personas():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, nombre, apellido, email, embedding FROM personas")
        return cur.fetchall()

def insert_deteccion(persona_id, emocion, confianza):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO detecciones (persona_id, emocion, confianza)
        VALUES (?, ?, ?)
        """, (persona_id, emocion, confianza))
        conn.commit()
        return cur.lastrowid